
import streamlit as st
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px 

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA


from sklearn_extra.cluster import KMedoids
from sklearn.cluster import OPTICS
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN


@st.cache_data
def load_data(data):
	df = pd.read_csv(data)
	return df

def variabel_input(data):
	v_in= data[['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN']]
	return v_in

def minmax_scaler(data):
	scaler = MinMaxScaler()
	data_mm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
	return data_mm

def pca_scale(data):
	pca_model = PCA(n_components=2)
	pca_vin = pca_model.fit_transform(data)
	hasil_pca = pd.DataFrame(pca_vin, columns=['Feature A', 'Feature B'])
	return hasil_pca

def bikin_dataframe(list1, list2,list3):
	series1 = pd.Series(list1)
	series2 = pd.Series(list2)
	series3 = pd.Series(list3)
	df = pd.concat([series1, series2, series3], axis=1)
	return df


def run_cl_app():
	data_awal = load_data('data/data_fix.csv')
	v_in=variabel_input(data_awal)
	vin_scale=minmax_scaler(v_in)
	hasil_pca=pca_scale(vin_scale)
	submenu = st.sidebar.selectbox("Submenu",['Model Awal','Evaluasi'])
	if submenu == 'Model Awal':
		
		st.write('Model Klastering Awal')

		silhouette_list=[]
		dbi_list=[]

		with st.expander('Tabel'):
			st.dataframe(vin_scale)


		with st.expander('KMedoids'):
			
			kmedoids=KMedoids(n_clusters=5, random_state=0).fit(vin_scale)
			y_kmed=kmedoids.fit_predict(vin_scale)
			silhouette_avg=silhouette_score(vin_scale,y_kmed)
			silhouette_list.append(silhouette_avg)
			dbi_kmed = davies_bouldin_score(vin_scale, y_kmed)
			dbi_list.append(dbi_kmed)
			st.write('Nilai Silhouette :{}, Nilai DBI :{}'.format(silhouette_avg,dbi_kmed))
			hasil_pca['KMEDOIDS']=y_kmed
			

			fig, ax = plt.subplots()
			scatter = ax.scatter(hasil_pca['Feature A'], hasil_pca['Feature B'], c=hasil_pca['KMEDOIDS'])
			legend = ax.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
			ax.add_artist(legend)
			st.pyplot(fig)

		with st.expander('OPTICS'):
			optics_model = OPTICS(min_samples=2, xi=0.5)

			# Fit the model1
			optics_model.fit(vin_scale)

			# Predict the clusters
			labels_optic = optics_model.labels_

			# Calculate the silhouette score
			silhouette_optics = silhouette_score(vin_scale, labels_optic)
			silhouette_list.append(silhouette_optics)

			dbi_opt = davies_bouldin_score(vin_scale, labels_optic)
			dbi_list.append(dbi_opt)

			hasil_pca['OPTICSS']=labels_optic
			st.write('Nilai Silhouette',silhouette_optics)

			fig, ax = plt.subplots()
			scatter = ax.scatter(hasil_pca['Feature A'], hasil_pca['Feature B'], c=hasil_pca['OPTICSS'])
			legend = ax.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
			ax.add_artist(legend)
			st.pyplot(fig)
	    
		with st.expander('Fuzzy C-Means'):
			#k = 5
			#m = 2.0
			x=vin_scale.values
			cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(x.T, 5, 2.0, error=0.005, maxiter=1000)
			labels_fuzz = np.argmax(u, axis=0)
			siluete_fcm = silhouette_score(x, labels_fuzz)
			
			st.write('Nilai Silhouette :',siluete_fcm)

			silhouette_list.append(siluete_fcm)
			hasil_pca['FCM']=labels_fuzz

			dbi_fcm = davies_bouldin_score(x, labels_fuzz)
			dbi_list.append(dbi_fcm)

			fig, ax = plt.subplots()
			scatter = ax.scatter(hasil_pca['Feature A'], hasil_pca['Feature B'], c=hasil_pca['FCM'])
			legend = ax.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
			ax.add_artist(legend)
			st.pyplot(fig)

		with st.expander('DBSCAN'):
			db = DBSCAN(eps=0.15, min_samples=5).fit(x)
			labels_db=db.labels_
			silhouette_score_dbscan = silhouette_score(x, db.labels_)
			
			st.write('Nilai Silhouette :', silhouette_score_dbscan)

			silhouette_list.append(silhouette_score_dbscan)

			dbi_dbs = davies_bouldin_score(x, labels_db)
			dbi_list.append(dbi_dbs)

			hasil_pca['DBSCAN']=labels_db

			fig, ax = plt.subplots()
			scatter = ax.scatter(hasil_pca['Feature A'], hasil_pca['Feature B'], c=hasil_pca['DBSCAN'])
			legend = ax.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
			ax.add_artist(legend)
			st.pyplot(fig)

		with st.expander('Hasil'):
			judul=['KMedoids','OPTICS','FCM','DBSCAN']
			rekap_sil=bikin_dataframe(judul,silhouette_list,dbi_list)
			rekap_sil.columns=['Algoritma','Silhouette','DBI']
			st.dataframe(rekap_sil.iloc[:,:2])

	if submenu == 'Evaluasi':
		st.write('Evaluasi')

		with st.expander('K Medoids'):
			st.write('K Medoids')


		with st.expander('Fuzzy C-Means'):
			st.write('Fuzzy C-Means')