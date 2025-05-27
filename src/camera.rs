use anyhow::Result;
use opencv::{
    prelude::*,
    videoio,
    highgui,
};

pub struct Camera {
    capture: videoio::VideoCapture,
    window_name: String,
}

impl Camera {
    pub fn new(camera_id: i32) -> Result<Self> {
        let mut capture = videoio::VideoCapture::new(camera_id, videoio::CAP_ANY)?;
        if !capture.is_opened()? {
            anyhow::bail!("Error: Could not open camera");
        }
        
        Ok(Self {
            capture,
            window_name: "Camera Feed".to_string(),
        })
    }

    pub fn capture_frame(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        self.capture.read(&mut frame)?;
        
        if frame.empty() {
            anyhow::bail!("Error: Empty frame");
        }
        
        Ok(frame)
    }

    pub fn display_frame(&self, frame: &Mat) -> Result<()> {
        highgui::named_window(&self.window_name, highgui::WINDOW_AUTOSIZE)?;
        highgui::imshow(&self.window_name, frame)?;
        Ok(())
    }

    pub fn should_quit(&self) -> Result<bool> {
        let key = highgui::wait_key(1)?;
        Ok(key == 'q' as i32)
    }

    pub fn cleanup(&self) -> Result<()> {
        highgui::destroy_all_windows()?;
        Ok(())
    }
}
