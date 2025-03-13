use std::fmt;
use opencv::core::Mat;

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug)]
pub struct TextBox {
    pub points: Vec<Point>,
    pub score: f32,
}

impl fmt::Display for TextBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TextBox[score({}), [x: {}, y: {}], [x: {}, y: {}], [x: {}, y: {}], [x: {}, y: {}]]",
            self.score,
            self.points[0].x, self.points[0].y,
            self.points[1].x, self.points[1].y,
            self.points[2].x, self.points[2].y,
            self.points[3].x, self.points[3].y,
        )
    }
}

#[derive(Debug, Default)]
pub struct Angle {
    pub index: i32,
    pub score: f32,
    pub time: f32,
}

impl fmt::Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header = if self.index >= 0 { "Angle" } else { "AngleDisabled" };
        write!(
            f,
            "{}[Index({}), Score({}), Time({}ms)]",
            header, self.index, self.score, self.time
        )
    }
}

#[derive(Debug, Default)]
pub struct TextLine {
    pub text: String,
    pub char_scores: Vec<f32>,
    pub time: f32,
}

impl fmt::Display for TextLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let scores = self.char_scores
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(",");
        write!(
            f,
            "TextLine[Text({}),CharScores({}),Time({}ms)]",
            self.text, scores, self.time
        )
    }
}

#[derive(Debug)]
pub struct TextBlock {
    pub box_points: Vec<Point>,
    pub box_score: f32,
    pub angle_index: i32,
    pub angle_score: f32,
    pub angle_time: f32,
    pub text: String,
    pub char_scores: Vec<f32>,
    pub crnn_time: f32,
    pub block_time: f32,
}

impl fmt::Display for TextBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = String::from("├─TextBlock\n");
        
        // TextBox
        output.push_str(&format!(
            "│   ├──TextBox[score({}), [x: {}, y: {}], [x: {}, y: {}], [x: {}, y: {}], [x: {}, y: {}]]\n",
            self.box_score,
            self.box_points[0].x, self.box_points[0].y,
            self.box_points[1].x, self.box_points[1].y,
            self.box_points[2].x, self.box_points[2].y,
            self.box_points[3].x, self.box_points[3].y,
        ));

        // Angle
        let header = if self.angle_index >= 0 { "Angle" } else { "AngleDisabled" };
        output.push_str(&format!(
            "│   ├──{}[Index({}), Score({}), Time({}ms)]\n",
            header, self.angle_index, self.angle_score, self.angle_time
        ));

        // TextLine
        let scores = self.char_scores
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(",");
        output.push_str(&format!(
            "│   ├──TextLine[Text({}),CharScores({}),Time({}ms)]\n",
            self.text, scores, self.crnn_time
        ));

        // BlockTime
        output.push_str(&format!("│   └──BlockTime({}ms)", self.block_time));

        write!(f, "{}", output)
    }
}

#[derive(Debug)]
pub struct OcrResult {
    pub text_blocks: Vec<TextBlock>,
    pub db_net_time: f32,
    pub box_img: Mat,
    pub detect_time: f32,
    pub str_res: String,
}

impl fmt::Display for OcrResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = String::from("OcrResult\n");
        
        // TextBlocks
        for block in &self.text_blocks {
            output.push_str(&block.to_string());
            output.push('\n');
        }

        output.push_str(&format!("├─DbNetTime({}ms)\n", self.db_net_time));
        output.push_str(&format!("├─DetectTime({}ms)\n", self.detect_time));
        output.push_str(&format!("└─StrRes({})", self.str_res));

        write!(f, "{}", output)
    }
} 