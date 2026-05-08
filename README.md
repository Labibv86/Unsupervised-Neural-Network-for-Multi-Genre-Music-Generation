

Contributions

Md. Shihab Sharar(22241028) --> Task 2,3,4,Evaluation Metrices 
Md. Moazzem Hossain Majumder(24241132) --> Preprocessing, Task 1

dataset--> Lakh MIDI Dataset
!!!!!!!!!!! Due to size issues in GITHUB we have uploaded the raw and preprocessed data in Google Drive.The whole project structure can also be found there.
Link: https://drive.google.com/drive/folders/1Ud_ujpi8qtAehqjmfW2TOi7_rf22lgnj?usp=sharing 




Project Structure

music_genesis/

data/raw_midi     we have placed Lakh MIDI dataset here
data/processed     the preprocessed tokenized outputs are here

src/preprocessing/    tokenization and preprocessing scripts   
src/models        Autoencoder,vae,transformer scripts are here
src/training        these are the model training python files are here
src/generation       sample generation scripts
src/evaluation      evaluation scripts of every model
src/rlhf         reinforcement learning folder containing the mock_reward, reward_funtion, human_survey,fine_tuning scripts
src/baselines       randomnotgenarator and Markovchain baseline model scripts
   
outputs        Generated samples, Evaluation matrices diagrams, plots and midi files.
shortcuts    we have monitored and saved all the model training  and sample generation performances
report           latex report files and PDF
EDA         Explanatory Data Analysis Diagrams      
requirements.txt        the libraries that we have imported to create the local environemt.
README.md


Commands

preprocess dataset: python src/preprocessing/tokenize_dataset.py
LSTM Autoencoder: python src/training/train_autoencoder.py
VAE: python src/training/train_vae.py
Transformer: python src/training/train_transformer.py

RLHF
python src/rlhf/human_survey.py
python src/rlhf/mock_ratings.py
python src/rlhf/reward_model.py
python src/rlhf/rlhf_finetune.py
python src/rlhf/compare_results.py


Generate samples
python src/generation/generate_music.py
python src/generation/generate_vae_samples.py
python src/generation/generate_transformer_samples.py

Evaluation Matrices
python src/evaluation/metrics_fixed.py
python src/baselines/baseline_models.py