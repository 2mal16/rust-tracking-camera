use std::arch::aarch64::uint16x4_t;

use anyhow::Result;
use opencv::{
    prelude::*,
    videoio,
    imgcodecs,
    core,
    imgproc,
};

struct ASD {
    size: uint16x4_t,
}

pub struct MotionDetector {
    previous_frame: Option<Mat>,
    min_area: f64,
}

impl MotionDetector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            previous_frame: None,
            min_area: 500.0, // Minimum area to consider as motion
        })
    }

    pub fn detect_motion(&mut self, current_frame: &Mat) -> Result<Vec<core::Rect>> {
        // Convert to grayscale
        let mut gray = Mat::default();
        imgproc::cvt_color(current_frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Apply Gaussian blur
        let mut blurred = Mat::default();
        imgproc::gaussian_blur(&gray, &mut blurred, core::Size::new(21, 21), 0.0, 0.0, core::BORDER_DEFAULT)?;

        // If this is the first frame, store it and return empty vector
        if self.previous_frame.is_none() {
            self.previous_frame = Some(blurred.clone());
            return Ok(Vec::new());
        }

        // Compute absolute difference between current and previous frame
        let mut frame_delta = Mat::default();
        core::absdiff(&self.previous_frame.as_ref().unwrap(), &blurred, &mut frame_delta)?;

        // Threshold the delta image
        let mut thresh = Mat::default();
        imgproc::threshold(&frame_delta, &mut thresh, 25.0, 255.0, imgproc::THRESH_BINARY)?;

        // Dilate the thresholded image
        let mut dilated = Mat::default();
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(3, 3),
            core::Point::new(-1, -1),
        )?;
        imgproc::dilate(&thresh, &mut dilated, &kernel, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        // Find contours
        let mut contours = opencv::types::VectorOfMat::new();
        let mut hierarchy = Mat::default();
        imgproc::find_contours(&dilated, &mut contours, &mut hierarchy, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;

        // Find bounding boxes for significant motion
        let mut bounding_boxes = Vec::new();
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            
            if area > self.min_area {
                let rect = imgproc::bounding_rect(&contour)?;
                bounding_boxes.push(rect);
            }
        }

        // Update previous frame
        self.previous_frame = Some(blurred);

        Ok(bounding_boxes)
    }

    pub fn draw_bounding_boxes(&self, frame: &Mat, boxes: &[core::Rect]) -> Result<Mat> {
        let mut result = frame.clone();
        
        for rect in boxes {
            imgproc::rectangle(
                &mut result,
                rect,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        Ok(result)
    }
}

fn main() -> Result<()> {
    // Open the default camera
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    
    // Check if camera opened successfully
    if !camera.is_opened()? {
        anyhow::bail!("Error: Could not open camera");
    }

    println!("Camera opened successfully!");
    println!("Press 'q' to quit");

    loop {
        let mut frame = Mat::default();
        camera.read(&mut frame)?;

        if frame.empty() {
            println!("Error: Empty frame");
            break;
        }

        // Display the frame
        let window_name = "Camera Feed";
        opencv::highgui::named_window(window_name, opencv::highgui::WINDOW_AUTOSIZE)?;
        opencv::highgui::imshow(window_name, &frame)?;

        // Break the loop if 'q' is pressed
        let key = opencv::highgui::wait_key(1)?;
        if key == 'q' as i32 {
            break;
        }
    }

    // Clean up
    opencv::highgui::destroy_all_windows()?;
    Ok(())
}
