import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) #allows to import CogBench as a package
from CogBench.base_classes import StoringScores

class StoringMetaCognitiveScores(StoringScores):
    def __init__(self):
        super().__init__()
        self.add_arguments_()

    def add_arguments_(self):
        # Add any additional arguments here
        self.parser.add_argument('--columns', nargs='+', default=['performance_score2','performance_score2_name','behaviour_score2','behaviour_score2_name','behaviour_score3','behaviour_score3_name'])

    def get_scores(self, df, storing_df, engine, run):
        """Get the scores for the metacognition task.
        Args:
            df (pd.DataFrame): Dataframe with the results of the experiment
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
            engine (str): Name of the engine used
            run (int): Number of the run
        Returns:
            storing_df (pd.DataFrame): Dataframe with the scores of the experiment
        """
        # Behaviour score 1: QSR
        qsr_s = 1 - (df['accurate'] - df['confidence'])**2
        qsr = qsr_s.mean()
        # Behaviour score 2: Confidence
        confidence = df['confidence'].mean()

        # Behaviour score 3: Adjusted QSR
        lb = df['confidence'].min()
        hb = df['confidence'].max()
        if hb-lb != 0:
            confidence_scaled = (df['confidence'] - lb) / (hb - lb)
            adjusted_qsr = 1 - (df['accurate'] - confidence_scaled)**2
            adjusted_qsr = adjusted_qsr.mean()
        else:
            adjusted_qsr = qsr

        # Performance score 1: Rewards
        rewards = df['reward'].mean()

        # Performance score 2: Mean accuracy
        mean_acc = df['accurate'].mean()

        # Add the final score to the csv file
        # if engine, run exists already in storing_df then update the values
        if  ((storing_df['engine'] == engine) & (storing_df['run'] == run)).any():
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score1'] = rewards
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score1'] = qsr
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score2'] = confidence
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'performance_score2'] = mean_acc
            storing_df.loc[(storing_df['engine'] == engine) & (storing_df['run'] == run), 'behaviour_score3'] = adjusted_qsr           
        else:
            storing_df.loc[len(storing_df)] = [engine, run, rewards, 'rewards mean', qsr, 'meta-cognitive sensitivity', confidence, 'confidence mean', mean_acc, 'mean accuracy', adjusted_qsr, 'adjusted qsr']
        return storing_df

if __name__ == '__main__':
    StoringMetaCognitiveScores().get_all_scores()