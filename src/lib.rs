use anyhow::Result;
use nom::{bytes::complete::take_until, IResult};

use bitvec_helpers::bitvec_reader::BitVecReader;

pub mod hevc;
pub mod utils;

use hevc::*;
use pps::PPSNAL;
use slice::SliceNAL;
pub use sps::SPSNAL;
use vps::VPSNAL;

use utils::clear_start_code_emulation_prevention_3_byte;

// We don't want to parse large slices because the memory is copied
const MAX_PARSE_SIZE: usize = 2048;

const HEADER_LEN_3: usize = 3;
const HEADER_LEN_4: usize = 4;
const NAL_START_CODE_3: &[u8] = &[0, 0, 1];
const NAL_START_CODE_4: &[u8] = &[0, 0, 0, 1];

pub enum NALUStartCode {
    Length3,
    Length4,
}

#[derive(Default)]
pub struct HevcParser {
    reader: BitVecReader,
    pub nalu_start_code: NALUStartCode,

    nals: Vec<NALUnit>,
    incomplete_nals: Vec<u8>,
    vps: Vec<VPSNAL>,
    sps: Vec<SPSNAL>,
    pps: Vec<PPSNAL>,
    ordered_frames: Vec<Frame>,
    frames: Vec<Frame>,

    latest_key_frame: Vec<u8>,
    buffering: bool,

    poc: u64,
    poc_tid0: u64,

    current_frame: Frame,
    decoded_index: u64,
    presentation_index: u64,
}

impl HevcParser {
    fn take_until_nal<'a>(tag: &[u8], data: &'a [u8]) -> IResult<&'a [u8], &'a [u8]> {
        take_until(tag)(data)
    }

    pub fn get_offsets(&mut self, data: &[u8], offsets: &mut Vec<usize>) {
        offsets.clear();

        let mut consumed = 0;

        let nal_start_tag = match &self.nalu_start_code {
            NALUStartCode::Length3 => NAL_START_CODE_3,
            NALUStartCode::Length4 => NAL_START_CODE_4,
        };

        loop {
            match Self::take_until_nal(nal_start_tag, &data[consumed..]) {
                Ok(nal) => {
                    // Byte count before the NAL is the offset
                    consumed += nal.1.len();

                    offsets.push(consumed);

                    // nom consumes the tag, so add it back
                    consumed += match &self.nalu_start_code {
                        NALUStartCode::Length3 => HEADER_LEN_3,
                        NALUStartCode::Length4 => HEADER_LEN_4,
                    };
                }
                _ => {
                    self.incomplete_nals.clear();

                    if data[consumed - 3..consumed] == [0, 0, 1] {
                        self.incomplete_nals.extend_from_slice(&[0, 0, 1]);
                    }

                    self.incomplete_nals.extend_from_slice(&data[consumed..]);
                    return;
                }
            }
        }
    }

    pub fn split_nals(
        &mut self,
        data: &[u8],
        offsets: &[usize],
        last: usize,
        parse_nals: &[u8],
    ) -> Result<Vec<NALUnit>> {
        let count = offsets.len();

        let mut nals = Vec::with_capacity(count);

        for (index, offset) in offsets.iter().enumerate() {
            let size = if offset == &last {
                data.len() - offset
            } else {
                let size = if index == count - 1 {
                    last - offset
                } else {
                    offsets[index + 1] - offset
                };

                match &data[offset + size - 1..offset + size + 3] {
                    [0, 0, 0, 1] => size - 1,
                    _ => size,
                }
            };

            let nal = self.parse_nal(data, *offset, size, parse_nals)?;

            nals.push(nal);
        }

        Ok(nals)
    }

    fn parse_nal(
        &mut self,
        data: &[u8],
        offset: usize,
        size: usize,
        parse_nal: &[u8],
    ) -> Result<NALUnit> {
        let mut nal = NALUnit::default();

        // Assuming [0, 0, 1] header
        // Offset is at first element
        let pos = offset + HEADER_LEN_3;
        let end = offset + size;

        let parsing_end = if size > MAX_PARSE_SIZE {
            offset + MAX_PARSE_SIZE
        } else {
            end
        };

        nal.start = pos;
        nal.end = end;
        nal.decoded_frame_index = self.decoded_index;

        nal.start_code_len = if offset > 0 {
            // Previous byte is 0, offset..offset + 3 is [0, 0, 1]
            // Actual start code is length 4
            if data[offset - 1] == 0 {
                4
            } else {
                3
            }
        } else {
            3
        };
        nal.nal_type = data[pos] >> 1;

        let should_parse = parse_nal.contains(&nal.nal_type);

        if should_parse {
            let bytes = clear_start_code_emulation_prevention_3_byte(&data[pos..parsing_end]);
            self.reader = BitVecReader::new(bytes);

            self.parse_nal_header(&mut nal)?;
        }

        if nal.nuh_layer_id > 0 {
            return Ok(nal);
        }

        if should_parse {
            match nal.nal_type {
                NAL_VPS => self.parse_vps()?,
                NAL_SPS => self.parse_sps()?,
                NAL_PPS => self.parse_pps()?,

                NAL_TSA_N | NAL_TSA_R | NAL_STSA_N | NAL_STSA_R | NAL_BLA_W_LP | NAL_BLA_W_RADL
                | NAL_BLA_N_LP | NAL_IDR_W_RADL | NAL_IDR_N_LP | NAL_CRA_NUT | NAL_RADL_N
                | NAL_RADL_R | NAL_RASL_N | NAL_RASL_R | NAL_IRAP_VCL22 | NAL_IRAP_VCL23 => {
                    self.parse_slice(&mut nal)?;

                    self.current_frame.nals.push(nal.clone());
                }
                NAL_SEI_SUFFIX | NAL_UNSPEC62 | NAL_UNSPEC63 => {
                    // Dolby NALs are suffixed to the slices
                    self.current_frame.nals.push(nal.clone());
                }
                _ => {
                    self.add_current_frame();

                    nal.decoded_frame_index = self.decoded_index;
                    self.current_frame.nals.push(nal.clone());
                }
            };

            // Parameter sets also mean a new frame
            match nal.nal_type {
                NAL_VPS | NAL_SPS | NAL_PPS => {
                    self.add_current_frame();

                    nal.decoded_frame_index = self.decoded_index;
                    self.current_frame.nals.push(nal.clone());

                    if self.buffering {
                        if nal.nal_type == NAL_VPS {
                            self.latest_key_frame.clear();

                            // Restrict from growing infintly
                            self.nals.clear();
                        }

                        self.latest_key_frame
                            .extend_from_slice(&data[nal.start - 3..nal.end]);
                    }
                }
                NAL_BLA_W_LP | NAL_BLA_W_RADL | NAL_BLA_N_LP | NAL_IDR_W_RADL | NAL_IDR_N_LP
                | NAL_CRA_NUT | NAL_IRAP_VCL22 | NAL_IRAP_VCL23 => {
                    if self.buffering {
                        self.latest_key_frame
                            .extend_from_slice(&data[nal.start - 3..nal.end]);
                        self.buffering = false;
                    }
                }
                _ => (),
            };

            self.nals.push(nal.clone());
        }

        Ok(nal)
    }

    fn parse_nal_header(&mut self, nal: &mut NALUnit) -> Result<()> {
        // forbidden_zero_bit
        self.reader.get()?;

        nal.nal_type = self.reader.get_n(6);

        if self.reader.available() < 9 && matches!(nal.nal_type, NAL_EOS_NUT | NAL_EOB_NUT) {
        } else {
            nal.nuh_layer_id = self.reader.get_n(6);
            nal.temporal_id = self.reader.get_n::<u8>(3) - 1;
        }

        Ok(())
    }

    fn parse_vps(&mut self) -> Result<()> {
        let vps = VPSNAL::parse(&mut self.reader)?;

        self.remove_vps(&vps);

        self.vps.push(vps);

        Ok(())
    }

    fn parse_sps(&mut self) -> Result<()> {
        let sps = SPSNAL::parse(&mut self.reader)?;
        self.remove_sps(&sps);

        self.sps.push(sps);

        Ok(())
    }

    fn parse_pps(&mut self) -> Result<()> {
        let pps = PPSNAL::parse(&mut self.reader)?;

        self.remove_pps(&pps);

        self.pps.push(pps);

        Ok(())
    }

    fn parse_slice(&mut self, nal: &mut NALUnit) -> Result<()> {
        let slice = SliceNAL::parse(
            &mut self.reader,
            &self.sps,
            &self.pps,
            nal,
            &mut self.poc_tid0,
            &mut self.poc,
        )?;

        // Consecutive slice NALs cases
        if self.current_frame.first_slice.first_slice_in_pic_flag && slice.first_slice_in_pic_flag {
            nal.decoded_frame_index = self.decoded_index + 1;
            self.add_current_frame();
        }

        if slice.key_frame {
            self.reorder_frames();
        }

        if slice.first_slice_in_pic_flag {
            self.current_frame.first_slice = slice;

            self.current_frame.decoded_number = self.decoded_index;
        }

        Ok(())
    }

    fn remove_vps(&mut self, vps: &VPSNAL) {
        let id = vps.vps_id as usize;

        if let Some(existing_vps) = self.vps.get(id) {
            if existing_vps == vps {
                self.vps.remove(id);

                let sps_to_remove: Vec<SPSNAL> = self
                    .sps
                    .clone()
                    .into_iter()
                    .filter(|sps| sps.vps_id == vps.vps_id)
                    .collect();

                sps_to_remove.iter().for_each(|sps| self.remove_sps(sps));
            }
        }
    }

    fn remove_sps(&mut self, sps: &SPSNAL) {
        let id = sps.sps_id as usize;

        if let Some(existing_sps) = self.sps.get(id) {
            if existing_sps == sps {
                self.sps.remove(id);

                // Remove all dependent pps
                self.pps.retain(|pps| pps.sps_id != sps.sps_id);
            }
        }
    }

    fn remove_pps(&mut self, pps: &PPSNAL) {
        // Remove if same id
        if let Some(existing_pps) = self.pps.get(pps.pps_id as usize) {
            if existing_pps == pps {
                self.pps.remove(pps.pps_id as usize);
            }
        }
    }

    // If we're here, the last slice of a frame was found already
    fn add_current_frame(&mut self) {
        if self.current_frame.first_slice.first_slice_in_pic_flag {
            self.decoded_index += 1;

            self.current_frame.presentation_number =
                self.current_frame.first_slice.output_picture_number;

            self.current_frame.frame_type = self.current_frame.first_slice.slice_type;

            self.frames.push(self.current_frame.clone());

            self.current_frame = Frame::default();
        }
    }

    fn reorder_frames(&mut self) {
        let mut offset = self.presentation_index;

        self.frames.sort_by_key(|f| f.presentation_number);
        self.frames.iter_mut().for_each(|f| {
            f.presentation_number = offset;
            offset += 1;
        });

        self.presentation_index = offset;
        self.ordered_frames.clear();
        self.ordered_frames.extend_from_slice(&self.frames);
        self.frames.clear();
    }

    pub fn display(&self) {
        println!("{} frames", &self.ordered_frames.len());
        for frame in &self.ordered_frames {
            let pict_type = match frame.frame_type {
                2 => "I",
                1 => "P",
                0 => "B",
                _ => "",
            };

            println!(
                "{} display order {} poc {} pos {}",
                pict_type,
                frame.presentation_number,
                frame.first_slice.output_picture_number,
                frame.decoded_number
            );
        }
    }

    pub fn finish(&mut self) {
        self.add_current_frame();
        self.reorder_frames();
    }

    /// Processed frames in the current GOP
    /// Cleared every key frame
    pub fn processed_frames(&self) -> &Vec<Frame> {
        &self.frames
    }

    pub fn ordered_frames(&self) -> &Vec<Frame> {
        &self.ordered_frames
    }

    pub fn get_nals(&self) -> &Vec<NALUnit> {
        &self.nals
    }

    pub fn parse_nalunits(&mut self, data: &[u8], parse_nals: &[u8]) {
        let mut offsets = vec![];
        // Prepend data with incomplete nals

        let new_len = data.len() + self.incomplete_nals.len();
        let mut new_data = Vec::with_capacity(new_len);
        new_data.extend_from_slice(self.incomplete_nals.as_slice());
        new_data.extend_from_slice(data);

        self.get_offsets(new_data.as_slice(), &mut offsets);

        if let Some(last_offset) = offsets.pop() {
            let _ = self.split_nals(new_data.as_slice(), &offsets, last_offset, parse_nals);
        }
    }

    pub fn get_sps(&self) -> &Vec<SPSNAL> {
        &self.sps
    }

    pub fn restart_buffering(&mut self) {
        self.buffering = true;
    }

    pub fn get_latest_key_frame(&self) -> Option<&[u8]> {
        if !self.buffering {
            Some(self.latest_key_frame.as_slice())
        } else {
            None
        }
    }
}

impl Default for NALUStartCode {
    fn default() -> Self {
        NALUStartCode::Length3
    }
}
