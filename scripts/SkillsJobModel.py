import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore",
                        message=".* was fitted with feature names.*")

class SkillsJobModel():
    def __init__(self, model_dict_path):
        # Load model
        self.model_dict_path = model_dict_path
        self.model_dict_path = model_dict_path
        self.model_dict = self.load_model_dict()
        self.validate_identical_features()

        # Extract features and target - skills & jobs
        self.jobs = list(self.model_dict.keys())
        self.features = pd.Series(self.model_dict[self.jobs[0]].feature_names_in_)

    ## ***  Initialization helper functions *** ##
    def load_model_dict(self):
        model_dict = pickle.load(open(self.model_dict_path, 'rb'))
        return model_dict

    def validate_identical_features(self):
        # Validate that all features are identical
        models_features = [list(model.feature_names_in_)
                           for job, model in self.model_dict.items()]
        all_identical = all(model_features == models_features[0]
                            for model_features in models_features)
        if not all_identical:
            raise Exception("Features are expected to be the same for all models")
        return True

    ## ***  Job Prediction methods *** ##
    def one_hot_encode(self, user_skills: list):
        # Validate
        skills_in_features = pd.Series(user_skills).isin(self.features)
        if not skills_in_features.all():
            missing_features = skills_in_features[skills_in_features == False].index.tolist()
            error_message = "Those skills are not a part of model: " + str(missing_features)
            raise Exception(error_message)

        # One hot encode
        ohe_skills = self.features.isin(user_skills).astype(int)
        return ohe_skills

    def predict_single_job_prob(self, user_skills: list, job: str):
        ohe_skills = self.one_hot_encode(user_skills)
        model = self.model_dict[job]
        prediction = model.predict_proba([ohe_skills])[0][1]
        return prediction

    def predict_jobs_probs(self, user_skills: list):
        predictions = {job: self.predict_single_job_prob(user_skills, job)
                       for job in self.jobs}
        predictions = pd.Series(predictions).sort_values(ascending=False)
        return predictions
