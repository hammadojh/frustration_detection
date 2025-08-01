Based on the content of `paper_features.csv`, here is a summary for each column with citations:

### **paper**
This column serves as the primary identifier for each research paper, typically including the lead author's last name and the publication year.
*   **Citation Example:** `Arunachalam et al (2001)`
*   **Citation Example:** `Hutto & Gilbert (2014)`
*   **Citation Example:** `Zhang et al (2025)`

### **predict_features**
This column lists features identified in the literature that are used to *predict* future user states like frustration or session dropout. Many papers focus on detection rather than prediction, so this column is often empty.
*   **`response_latency`**: The time a system takes to respond, which can signal future dissatisfaction, especially for experienced users (`Azaria et al (2023)`).
*   **`intent_switch_frequency`**: How often a user changes their goal, which can be predictive of their satisfaction with a recommender system (`Cai et al (2020)`).
*   **`session_dropout`**: Used to predict if a user will abandon their session early (`Hsu et al (2024)`, `Yang & Hsu (2021)`).
*   **`sentiment_trajectory`**: The progression of user sentiment over a conversation, used to forecast overall customer satisfaction (`Bodigutla et al (2020)`, `Chatterjee et al (2020)`).

### **detect_features**
This column contains features used to *detect* a user's current state of frustration, confusion, or dissatisfaction.
*   **`politeness_level`**: A decrease in politeness can indicate frustration. This is observed in adults (`Danescu-Niculescu-Mizil et al (2013)`) and children (`Arunachalam et al (2001)`, `Potamianos et al (2004)`).
*   **`alignment_failures`**: When the system's response does not align with the user's expectation or question, indicating a breakdown (`Balaraman et al (2023)`, `Li et al (2024)`).
*   **`typing_speed_change`**: Changes in typing speed and pressure can signal user stress (`Epp et al (2011)`, `Sun et al (2021)`).
*   **`intent_repetition`**: The user repeating their intent is a strong signal of system misunderstanding and user frustration (`Hough & Schlangen (2015)`, `Oraby et al (2017)`).
*   **`escalation_request`**: The user explicitly asking to speak to a human agent (`Hernandez Caralt et al. (2024/2025)`).
*   **`exclamation_density`**: An increased use of exclamation marks can indicate heightened emotion (`Hutto & Gilbert (2014)`, `Leonova & Zuters (2021)`).

### **download_status & download_link**
These columns track the availability of the paper's PDF. `download_status` is marked as `DOWNLOADED` or `MANUAL_DOWNLOAD_NEEDED`. The `download_link` provides a direct URL to the paper when available.

### **paper_year**
This column indicates the year of publication for each paper, ranging from `2001` (`Arunachalam et al (2001)`) to `2025` (`Fourier GNN MERC (2025 AAAI)`, `Hernandez Caralt et al. (2024/2025)`).

### **#detect_features & #predict_features**
These columns provide a simple count of the features listed in the `detect_features` and `predict_features` columns, respectively. For example, `Hough & Schlangen (2015)` lists `4` detection features and `0` prediction features.

### **training_dataset(s) & testing_dataset(s)**
These columns detail the datasets used to train and evaluate the models.
*   **Public Datasets**: Many studies use well-known academic datasets like `MELD`, `IEMOCAP` (`Ghosal et al (2019)`), and `ReDial` (`Cai et al (2020)`).
*   **Proprietary Datasets**: Some research, particularly from industry labs, uses large-scale internal data from platforms like Amazon Alexa (`Bodigutla et al (2020)`, `Park et al (2020)`).
*   **Custom Datasets**: Other papers rely on data collected from specific lab experiments (`Azaria et al (2023)`) or from scraping online sources like Twitter (`Begovic et al (2023)`).

### **architecture_summary**
This column offers a brief technical summary of the method or model proposed in each paper.
*   **`Azaria et al (2023)`**: "A between-subjects lab experiment (N=202) was conducted where participants interacted with a chatbot that responded either instantly or with a delay."
*   **`Hutto & Gilbert (2014)`**: "Introduces VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment analysis tool."
*   **`Li et al (2024)`**: "Introduces MultConDB, a multimodal model for detecting dialogue breakdown in real-time. The model uses both audio and text signals, leveraging Wav2Vec2 for acoustic features and RoBERTa for textual features."
*   **`Ghosal et al (2019)`**: "Proposes a Dialogue Graph Convolutional Network (DialogueGCN) to model emotion recognition in conversations."

### **results_summary**
This column summarizes the key findings and outcomes of each study.
*   **`Danescu-Niculescu-Mizil et al (2013)`**: Found that polite Wikipedia editors are more likely to achieve high status, but they become less polite after being promoted.
*   **`Epp et al (2011)`**: Showed that a large majority of participants (>79%) exhibited significantly increased typing pressure during stressful conditions.
*   **`Li et al (2024)`**: Concluded that "Multimodal models outperform text-only models in detecting dialogue breakdowns."
*   **`Azaria et al (2023)`**: Found that a delayed response time has opposing effects on users depending on their prior experience with chatbots.

### **limitations_summary**
This column highlights the limitations acknowledged by the authors or inherent in the study's design.
*   **Dataset Specificity**: Many findings are limited to the specific language, domain, or platform studied, such as the Bosnian language on Twitter (`Begovic et al (2023)`) or a proprietary healthcare dataset (`Li et al (2024)`).
*   **Proprietary Data**: The use of non-public, internal datasets limits the reproducibility of the research (`Bodigutla et al (2020)`, `Hsu et al (2024)`).
*   **Artificial Settings**: Studies conducted in controlled lab settings (`Azaria et al (2023)`) or using simulated dialogues (`Mehri & Eskenazi (2021)`) may not generalize to real-world scenarios.
*   **Subjectivity**: Annotations for user satisfaction are often based on Likert scales, which can be subjective (`Bodigutla et al. (2019/2020)`).