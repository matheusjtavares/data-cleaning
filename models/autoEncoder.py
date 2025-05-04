
import pandas as pd
import numpy as np
import seaborn as sns
import time as tm
import os
from scipy.stats import t
from scipy.stats import kstest
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from models.frozenCleaner import frozenCleaner
from models.outlierCleaner import outlierCleaner
from tensorflow.keras.regularizers import l1
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import json
class turbAI():

    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.data_folder = os.path.join(self.project_root,'input/')
        self.chart_output_path = os.path.join(self.project_root,'output/charts')
        self.fcleaner = frozenCleaner()
        self.ocleaner = outlierCleaner()
    
    def get_inputs_from_filename(self,extension:str='.csv',filename:str='',add_pivot = True):
        path = os.path.join(self.data_folder,filename)

        extensions_dict = {
            '.csv':pd.read_csv,
            '.xlsx':pd.read_excel
        }
        df = extensions_dict['.csv'](rf'{path}',sep=';')
        if add_pivot:
            df.VALUE = df.VALUE.str.replace(',','.').astype(float)
            df = df.dropna()
            df = df.pivot_table(values='VALUE',index=['TS'],columns='CHANNEL_REG_ID').reset_index()
        return df
    
    def convert_df_to_tensors(self,df,output_column:str = '',):
        X = tf.convert_to_tensor(df[df.columns[df.columns!=output_column]].values,tf.float32)
        Y = tf.convert_to_tensor(df[output_column].values,tf.float32)
        print(X,Y)
        return X,Y
    
    def build_and_generate_model(self,x_tensor,y_tensor,hidden_layers,epochs=5,learning_rate=0.1,loss='mse'):
        model = tf.keras.Sequential()
        
        # Add input layer separately
        input_layer = tf.keras.layers.Input(shape=(x_tensor.shape[1],))
        model.add(input_layer)

        sorted_layers = sorted(hidden_layers.items())
        # Add remaining layers from dictionary
        for index, layer in sorted_layers:
            model.add(layer)

        model.add(tf.keras.layers.Dense(x_tensor.shape[1], activation='linear'))
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',          # what to monitor
            patience=10,                 # how many epochs to wait for improvement
            restore_best_weights=True,  # go back to best model before val_loss worsened
            min_delta=0.0001,           # minimum change to qualify as improvement
            verbose=1                   # show when it stops
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate), loss=loss)
        history = model.fit(x_tensor, y_tensor, epochs=epochs, 
                  verbose=1,shuffle=True,validation_split=0.2,
                    callbacks=[early_stop])

        return model,history
    
    def build_multiple_from_dict(self,df,network_settings,desired_output):
        desired_output = desired_output
        X,Y = self.convert_df_to_tensors(df,output_column=desired_output)
        X = tf.convert_to_tensor(df.values,tf.float32)
        for index,network in network_settings.items():
            model,model_history = self.build_and_generate_model(x_tensor=X,y_tensor=X,hidden_layers=network['hidden_layers'],
                                          epochs=network['epochs'],learning_rate=network['learning_rate'],loss=network['loss'])
            with open("best_model_history.json", "w") as f:
                json.dump(model_history.history, f)
            if np.min(model_history.history['loss']) < 0.01:
                model.save(fr'saved_models/turbAI_auto_{network['loss']}_{round(np.min(model_history.history['loss']),5)}.keras')
        return np.min(model_history.history['loss']),model
    
    def train_model(self):
        downtime_df = self.get_inputs_from_filename(extension='.csv',filename= 'dn_ter_3.csv',add_pivot=False)
        downtime_df['TS_START'] = pd.to_datetime(downtime_df['TS_START']).dt.floor('10min')
        downtime_df['TS_END'] = pd.to_datetime(downtime_df['TS_END']).dt.ceil('10min')
        for column in ['EL_AMA','EL_NEIGHBOR','EL_POWER_CURVE','EL_WINDFARM_AVERAGE']:
            downtime_df[column] = downtime_df[column].str.replace(',','.').astype(float)
        downtime_df['TOTAL_LOSS'] = downtime_df[['EL_AMA','EL_NEIGHBOR','EL_POWER_CURVE','EL_WINDFARM_AVERAGE']].sum(axis=1)

        df = self.get_inputs_from_filename(extension='.csv',filename= 'TURBINE_TER_3_2024.csv')
        print(f'Total Lines raw df {df.shape[0]}')
        df = df.drop(columns=[10])
        df['TS'] = pd.to_datetime(df['TS'])
        df = df.dropna() 
        print(f'Total Lines na df {df.shape[0]}')
        for ts_start,ts_end in zip(downtime_df['TS_START'],downtime_df['TS_END']):
            date_list = pd.date_range(start=ts_start, end=ts_end, freq='10min')
            df = df[~df.TS.isin(date_list)]
        print(f'Total Lines downtime df {df.shape[0]}')
        df = df.loc[df[50]==60]
        print(f'Total Lines status df {df.shape[0]}')
        for column in df.columns:
            if column in ['TS',50]:
                continue
            outliers,time = self.fcleaner.frozen_by_threshold(target=column,df=df,ts_column='TS')
            df = df[~df.TS.isin(outliers.TS)]
        print(f'Total Lines unfrozen df {df.shape[0]}')
        for column in df.columns:
            if column in ['TS',50]:
                continue
            outliers,time = self.ocleaner.modified_z_score(target=column,df=df)
            df = df[~df.TS.isin(outliers.TS)]
        df = df.drop(columns=['Modified_Z_Score'])
        print(f'Total Lines Mzscore df {df.shape[0]}')
        df = df.drop(columns=[50])
        self.build_correlation_chart(df)
        normalization_dict = {}
        for column in df.columns:
            if column == 'TS':
                continue
            normalization_dict[column]={
                'max':df[column].max(),
                'min':df[column].min(),
                }
        with open('normalization.json', 'w') as fp:
            json.dump(normalization_dict, fp)
        normalization_dict
        normalized_df = df.copy()
        columns_to_normalize = normalized_df.columns.difference(['TS'])

        normalized_df[columns_to_normalize] = (
            normalized_df[columns_to_normalize] - normalized_df[columns_to_normalize].min()
        ) / (
            normalized_df[columns_to_normalize].max() - normalized_df[columns_to_normalize].min()
        )    
        fig, ax = plt.subplots(figsize=(8, 4))
        plt.figure(figsize=(15, 6)) 
        sns.boxplot(data=normalized_df,fliersize=0,ax=ax)
        ax.set_xlabel("")
        plt.xticks(rotation=90)  # Rotaciona os rótulos das colunas para melhor visualização
        plt.savefig(os.path.join(self.chart_output_path, "Normalized Inputs Box Plot.png"), dpi=600, bbox_inches='tight')
        desired_output = 20
        model_loss = 9999
        np.random.seed(42)
        normalized_df = normalized_df.drop(columns = ['TS'])
        while model_loss>=0.002:
            layers = np.random.randint(3, 5)
            hidden_layers = {
                x:tf.keras.layers.Dense(np.random.randint(50, 200), kernel_regularizer=l1(0.0001), activation='relu')
                for x in list(range(layers))
            }
            layers_0 = np.random.randint(60, 80)
            layers_1 = np.random.randint(30, 40)
            layers_2 = np.random.randint(15, 20)
            bneck = np.random.randint(10, layers_2)
            hidden_layers = {
                0:tf.keras.layers.Dense(layers_0, activation='relu'),
                1:tf.keras.layers.Dense(layers_1, activation='relu'),
                2:tf.keras.layers.Dense(layers_2, activation='relu'),
                3:tf.keras.layers.Dense(bneck, kernel_regularizer=l1(0.0001), activation='relu'),
                4:tf.keras.layers.Dense(layers_2, activation='relu'),
                5:tf.keras.layers.Dense(layers_1, activation='relu'),
                6:tf.keras.layers.Dense(layers_0, activation='relu'),
            }

            # hidden_layers = {
            #     0:tf.keras.layers.Dense(79, activation='relu'),
            #     1:tf.keras.layers.Dense(36, activation='relu'),
            #     2:tf.keras.layers.Dense(18, activation='relu'),
            #     3:tf.keras.layers.Dense(17, kernel_regularizer=l1(0.0001), activation='relu'),
            #     4:tf.keras.layers.Dense(18, activation='relu'),
            #     5:tf.keras.layers.Dense(36, activation='relu'),
            #     6:tf.keras.layers.Dense(79, activation='relu'),
            # }
            # learning_rate = 0.00017336464952677488
            loss = np.random.choice(['mae','mse'])
            learning_rate = np.random.uniform(0,0.001)
            network_settings = {
                0:{
                'hidden_layers':hidden_layers,
                'epochs' : 200,
                'loss': loss,
                'learning_rate':learning_rate
                }
            }
            try:
                model_loss,model = self.build_multiple_from_dict(df=normalized_df,network_settings=network_settings,desired_output=desired_output)
                test = self.test_autoencoder_model(model=model)
                if test:
                    break
            except:
                continue
        print('found a match!')
        return
    def load_normalized_df(self):
        raw_df = self.get_inputs_from_filename(extension='.csv',filename= 'TURBINE_TER_3_2024.csv')
        raw_df = raw_df.drop(columns=[10])
        raw_df.TS = pd.to_datetime(raw_df.TS)
        downtime_df = self.get_inputs_from_filename(extension='.csv',filename= 'dn_ter_3.csv',add_pivot=False)
        downtime_df['TS_START'] = pd.to_datetime(downtime_df['TS_START']).dt.floor('10min')
        downtime_df['TS_END'] = pd.to_datetime(downtime_df['TS_END']).dt.ceil('10min')
        for column in ['EL_AMA','EL_NEIGHBOR','EL_POWER_CURVE','EL_WINDFARM_AVERAGE']:
            downtime_df[column] = downtime_df[column].str.replace(',','.').astype(float)
        downtime_df['TOTAL_LOSS'] = downtime_df[['EL_AMA','EL_NEIGHBOR','EL_POWER_CURVE','EL_WINDFARM_AVERAGE']].sum(axis=1)
        downtime_df = downtime_df[~((downtime_df.ALARM_REG_ID == 712 )& (downtime_df.TOTAL_LOSS < 100))]
        # raw_df = raw_df.ffill()
        # raw_df = raw_df.dropna()  
        df = raw_df.copy()
        df = df.dropna()



        for ts_start,ts_end in zip(downtime_df['TS_START'],downtime_df['TS_END']):
            date_list = pd.date_range(start=ts_start, end=ts_end, freq='10min')
            df = df[~df.TS.isin(date_list)]
        df = df.loc[df[50]==60]   

        # for column in df.columns:
        #     if column in ['TS',50]:
        #         continue
        #     outliers,time = self.fcleaner.frozen_by_threshold(target=column,df=df,ts_column='TS')
        #     df = df[~df.TS.isin(outliers.TS)]
        for column in df.columns:
            if column in ['TS',50]:
                continue
            outliers,time = self.ocleaner.modified_z_score(target=column,df=df)
            df = df.drop(columns=['Modified_Z_Score'])
            df = df[~df.TS.isin(outliers.TS)]
        df = df.drop(columns=[50])

        raw_df = raw_df.drop(columns=[50])
        # raw_df = raw_df.ffill()
        raw_df = raw_df.fillna(0)
        
        raw_df['original_anomaly'] = raw_df.apply(lambda x: False if x.TS in df.TS.values else True,axis = 1)
        normalized_df = raw_df.copy()
        columns_to_normalize = normalized_df.columns.difference(['TS','original_anomaly'])
        with open('normalization.json', 'r') as fp:
            normalization_dict = json.load(fp)

        for col in columns_to_normalize:
            col_min = normalization_dict[str(col)]['min']
            col_max = normalization_dict[str(col)]['max']
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
        return normalized_df,columns_to_normalize
    
    def plot_model_results(self,model,normalized_df,columns_to_normalize):
        # Get precision, recall, and threshold values
        
         
        label_map = {True: "Anômalo", False: "Sadio"}
        anomaly_counts = normalized_df["original_anomaly"].map(label_map).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(anomaly_counts.index, anomaly_counts.values, color=["#EF5350", "#66BB6A"], edgecolor='black')
        # Style
        ax.set_xlabel("Quantidade", fontsize=11)
        ax.set_ylabel("Classe", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines[['right', 'top']].set_visible(False)  # cleaner look
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # Add value labels to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + max(anomaly_counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{int(width)}", va='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_output_path, "Anomaly_Division.png"), dpi=600, bbox_inches='tight')
        plt.close()
        

        X = tf.convert_to_tensor(normalized_df[columns_to_normalize].values,tf.float32)
        y_pred = model.predict(X)
        reconstruction_error = np.mean(np.square(normalized_df[columns_to_normalize].values - y_pred), axis=1)
        # Add to the DataFrame
        normalized_df["reconstruction_error"] = reconstruction_error
        precision, recall, thresholds = precision_recall_curve(normalized_df['original_anomaly'], reconstruction_error)
        # Calculate F1 score for each threshold
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]
        def find_closest_index(lst, target):
            return min(range(len(lst)), key=lambda i: abs(lst[i] - target))

        # Example
        numbers = [10, 22, 14, 3, 7, 17]
        target_precision= 0.85

        closest_index = find_closest_index(precision, target_precision)

        fig, ax = plt.subplots(figsize=(8, 5))
        precision_line = plt.plot(precision, label='Precisão')
        recall_line = plt.plot(recall, label='Recall') # cleaner look
        f1_line = plt.plot(f1, label='F1') # cleaner look
        min_precision = plt.axhline(target_precision, color='red', linestyle='--', label=f'85% Linha de Referência')
        lfo_line = plt.axvline(best_idx, color='green', linestyle='--', label=f'Max Score F1 - LFO = {thresholds[best_idx]:.4f}')
        target_line = plt.axvline(closest_index, color='salmon', linestyle='--', label=f'Max Recall 85% Precisão - LFO = {thresholds[closest_index]:.4f}')
        ax.set_ylabel("Value", fontsize=11)

        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_output_path,'Precision, Recall, F1'), dpi=600, bbox_inches='tight')
        plt.close()
        
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]

        print(f"Best Threshold F1 Based: {best_threshold:.6f}")
        print(f"F1 Score: {best_f1:.4f}")
        print(f"Recall: {recall[best_idx]:.4f}")
        print(f"Precision: {precision[best_idx]:.4f}")

        print(f"Best Threshold Precision 85%: {thresholds[closest_index]:.6f}")
        print(f"F1 Score: {f1[closest_index]:.4f}")
        print(f"Recall: {recall[closest_index]:.4f}")
        print(f"Precision: {precision[closest_index]:.4f}")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        precision_line = plt.plot(thresholds, precision[:-1], label='Precisão')
        recall_line = plt.plot(thresholds, recall[:-1], label='Recall')
        f1_line = plt.plot(thresholds, f1[:-1], label='F1 Score')
        bfo_line = plt.axvline(best_threshold, color='red', linestyle='--', label=f'LFO - {best_threshold:.4f}')
        ax.set_xlabel("Erro de Reconstrução (MSE)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.spines[['right', 'top']].set_visible(False)  # cleaner look
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_output_path,'Precision, Recall, F1 vs Threshold'), dpi=600, bbox_inches='tight')
        plt.close()


        threshold = thresholds[closest_index] 

        rec_errors = normalized_df["reconstruction_error"]  # or whatever column you're plotting

        # Create color list based on threshold
        colors = ["#EF5350" if val > threshold else "#66BB6A" for val in rec_errors]

        normalized_df["is_anomaly"] = (reconstruction_error > threshold)
        # normalized_df.loc[normalized_df["reconstruction_error"]>=1,"reconstruction_error"] = 0.01
        fig, ax = plt.subplots(figsize=(8, 4))
        
        error_dist = plt.scatter(normalized_df["TS"],rec_errors,c=colors, alpha=0.7, edgecolor='k', s=20)
        thresh_line = plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
        ax.set_xlabel("TS")
        ax.set_ylabel("Erro de Reconstrução (MSE)")
        ax.legend()
        ax.spines[['right', 'top']].set_visible(False)  # cleaner look
        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_output_path,'Distribution Across Errors'), dpi=600, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4))
        
        rec_range_errors = normalized_df[normalized_df.reconstruction_error <= 0.2]["reconstruction_error"]  # or whatever column you're plotting

        # Create color list based on threshold
        range_colors = ["#EF5350" if val > threshold else "#66BB6A" for val in rec_range_errors]

        error_dist = plt.scatter(normalized_df[normalized_df.reconstruction_error <= 0.2]["TS"],rec_range_errors,c=range_colors, alpha=0.7, s=20)
        thresh_line = plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
        ax.set_xlabel("TS")
        ax.set_ylabel("Erro de Reconstrução (MSE)")
        ax.legend()
        ax.spines[['right', 'top']].set_visible(False)  # cleaner look
        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_output_path,'Distribution Across Errors Range 1'), dpi=600, bbox_inches='tight')
        plt.close()
        
        

        # Create color list based on threshold
        fig, ax = plt.subplots(figsize=(8, 4))
        error_hist = plt.hist(normalized_df[normalized_df.reconstruction_error <= 0.2]['reconstruction_error'],bins = 200, alpha=0.7, edgecolor='k')
        thresh_line = plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
        ax.set_xlabel("Erro de Reconstrução (MSE)")
        ax.set_ylabel("Total de Amostras")
        ax.legend()
        ax.spines[['right', 'top']].set_visible(False)  # cleaner look
        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_output_path,'Error Histogram'), dpi=600, bbox_inches='tight')
        plt.close()
        

        fig, ax = plt.subplots(figsize=(8, 4))
        cm = confusion_matrix(normalized_df['original_anomaly'], normalized_df["is_anomaly"])
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax)
        # Customizations
        ax.set_title("")  # Remove the title
        ax.set_xlabel("Rotulação prevista")
        ax.set_ylabel("Rotulação alterada")

        plt.tight_layout()
        plt.savefig(os.path.join(self.chart_output_path,'Confusion Matrix'), dpi=600, bbox_inches='tight')
        plt.close()

        return

    def test_autoencoder_model(self,model=None,model_name=''):
        if model_name!='':
            model = tf.keras.models.load_model(fr'saved_models/{model_name}')
        raw_df = self.get_inputs_from_filename(extension='.csv',filename= 'TURBINE_TER_3_2024.csv')
        raw_df = raw_df.drop(columns=[10])
        raw_df.TS = pd.to_datetime(raw_df.TS)
        downtime_df = self.get_inputs_from_filename(extension='.csv',filename= 'dn_ter_3.csv',add_pivot=False)
        downtime_df['TS_START'] = pd.to_datetime(downtime_df['TS_START']).dt.floor('10min')
        downtime_df['TS_END'] = pd.to_datetime(downtime_df['TS_END']).dt.ceil('10min')
        for column in ['EL_AMA','EL_NEIGHBOR','EL_POWER_CURVE','EL_WINDFARM_AVERAGE']:
            downtime_df[column] = downtime_df[column].str.replace(',','.').astype(float)
        downtime_df['TOTAL_LOSS'] = downtime_df[['EL_AMA','EL_NEIGHBOR','EL_POWER_CURVE','EL_WINDFARM_AVERAGE']].sum(axis=1)
        downtime_df = downtime_df[~((downtime_df.ALARM_REG_ID == 712 )& (downtime_df.TOTAL_LOSS < 100))]
        raw_df = raw_df.ffill()
        df = raw_df.copy()
        for column in df.columns:
            if column in ['TS',50]:
                continue
            outliers,time = self.fcleaner.frozen_by_threshold(target=column,df=df,ts_column='TS')
            df = df[~df.TS.isin(outliers.TS)]
        
        for column in df.columns:
            if column in ['TS',50]:
                continue
            outliers,time = self.ocleaner.modified_z_score(target=column,df=df)
            df = df.drop(columns=['Modified_Z_Score'])
            df = df[~df.TS.isin(outliers.TS)]
        df = df.loc[df[50]==60]
        for ts_start,ts_end in zip(downtime_df['TS_START'],downtime_df['TS_END']):
            date_list = pd.date_range(start=ts_start, end=ts_end, freq='10min')
            df = df[~df.TS.isin(date_list)]
        df = df.drop(columns=[50])
        raw_df = raw_df.drop(columns=[50])

        raw_df['original_anomaly'] = raw_df.apply(lambda x: False if x.TS in df.TS.values else True,axis = 1)
        normalized_df = raw_df.copy()
        columns_to_normalize = normalized_df.columns.difference(['TS','original_anomaly'])
        with open('normalization.json', 'r') as fp:
            normalization_dict = json.load(fp)
        for col in columns_to_normalize:
            col_min = normalization_dict[str(col)]['min']
            col_max = normalization_dict[str(col)]['max']
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)

        X = tf.convert_to_tensor(normalized_df[columns_to_normalize].values,tf.float32)
        y_pred = model.predict(X)
        reconstruction_error = np.mean(np.square(normalized_df[columns_to_normalize].values - y_pred), axis=1)
        # Add to the DataFrame
        normalized_df["reconstruction_error"] = reconstruction_error


        # Get precision, recall, and threshold values
        precision, recall, thresholds = precision_recall_curve(normalized_df['original_anomaly'], reconstruction_error)

        # Calculate F1 score for each threshold
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]
        print(f"Best Threshold: {best_threshold:.6f}")
        print(f"Best F1 Score: {best_f1:.4f}")
        if best_f1 >= 0.853:
            return True
        else:
            return False
    def demo_model(self,model_name):
        model = tf.keras.models.load_model(fr'saved_models/{model_name}')
        normalized_df,columns_to_normalize = self.load_normalized_df()
        self.plot_model_results(model,normalized_df,columns_to_normalize)

if __name__=='__main__':
    turb = TurbAI()
    turb.train_model()
    # test_model()
    
    # test_autoencoder_model('best_model_copy.keras')
    test_autoencoder_model('turbAI_auto_f1_0.8177843306227798.keras')