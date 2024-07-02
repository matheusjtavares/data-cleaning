
from models.outlierCleaner import outlierCleaner
from models.dataPlotter import dataPlotter



data_plotter = dataPlotter(file_name='T1.csv')
outlier_cleaner = outlierCleaner(file_name='T1.csv')
outlier_cleaner.grubbs_test(target='Velocidade do Vento (m/s)')

data_plotter.make_all_histogram()
data_plotter.plot_time_series(width=25,height=10)

data_plotter.make_histogram(target='Velocidade do Vento (m/s)',bins=40)
data_plotter.make_histogram(target='Potência Teórica (kW)',bins=40)

outlier_cleaner.z_score(target='Velocidade do Vento (m/s)')
a,b=outlier_cleaner.modified_z_score(target='Velocidade do Vento (m/s)',limit=3)
outlier_cleaner.source_data.describe()
data_plotter.plot_time_series(width=25,height=10)



