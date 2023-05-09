import streamlit as st 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from joblib import load


@st.cache_data
def load_data(data):
	df = pd.read_csv(data)
	return df


def pilih_kolom(data):
	data_kolom=data[['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN', 
			'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI']]
	return data_kolom



def run_simulasi_app():
	st.write('Simulasi Prediksi')
		
	data_awal = load_data('data/data_fix.csv')

def minmax_scaler(data_awal,data_baru):
	scaler = MinMaxScaler()
	scaler.fit_transform(data_awal)
	scaled_new_data = scaler.transform(data_baru)
	return scaled_new_data

def load_model():
	model_gb=load('model/model_gb.joblib')
	return model_gb

def run_simulasi_app():

	data_awal = load_data('data/data_fix.csv')
	data_kolom = pilih_kolom(data_awal)
	data_baru=np.array([36,4297,31733,4922,7,30,7211834457,103.06,138.82,20.07,1463,1786,185])
	contoh=data_kolom.iloc[:1,:13]
	with st.expander('Data Awal'):
		#hasil_ts= minmax_scaler(data_kolom, data_kolom)
		st.dataframe(data_kolom)

	#with st.expander('Data Baru'):
	st.write('Variabel Input :')
	col1,col2,col3,col4=st.columns([1,1,1,1])

	with col1 :
		x1=st.number_input('WP_BENDAHARA',  value=36, min_value=0)
		x5=st.number_input('JUMLAH_FPP',value=7, min_value=0)
	with col2:
		x2=st.number_input('WP_BADAN',  value=4297, min_value=0)
		x6=st.number_input('JUMLAH_AR',  value=30, min_value=0)
		
	with col3:
		x3=st.number_input('WP_OP_KARYAWAN',  value=31733, min_value=0)
		x7=st.number_input('REALISASI_ANGGARAN',  value=7211834457, min_value=0)
		
	with col4:
		x4=st.number_input('WP_OP_PENGUSAHA',  value=4922, min_value=0)
		
	st.markdown("<hr>", unsafe_allow_html=True)
	st.write('Variabel Output :')
	col1,col2,col3=st.columns([1,1,1])
	with col1:
		x8=st.number_input('KEPATUHAN_SPT',  value=103.06, min_value=0.0)
		x11=st.number_input('SP2DK_TERBIT',  value=1990, min_value=0)
		
	with col2:
		x9=st.number_input('CAPAIAN_PENERIMAAN',  value=138.82, min_value=0.0)
		x12=st.number_input('SP2DK_CAIR',  value=1786, min_value=0)
		
	with col3:
		x10=st.number_input('PERTUMBUHAN_PENERIMAAN',  value=20.07)
		x13=st.number_input('PEMERIKSAAN_SELESAI',  value=185, min_value=0)
		
	st.markdown("<hr>", unsafe_allow_html=True)
	
	col1,col2=st.columns([1,1])	
	with col1:
		if st.button('Proses'):
			data = {'kolom_1': x1,
					'kolom_2': x2,
					'kolom_3': x3,
					'kolom_4': x4,
					'kolom_5': x5,
			        'kolom_6': x6,
			        'kolom_7': x7,
			        'kolom_8': x8,
			        'kolom_9': x9,
			        'kolom_10': x10,
			        'kolom_11': x11,
			        'kolom_12': x12,
			        'kolom_13': x13}
				# data_frame=pd.DataFrame(data=data, columns=['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN', 
				# 'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI'])
				# #st.write(data))
			df_input=pd.DataFrame(data,index=[0, 1, 2,3,5,6,7,8,9,10,11,12])
			model=load('model/model_mlp.joblib')
				
			hasil_minmax=minmax_scaler(data_kolom,df_input)
			hasil_predict=model.predict(hasil_minmax)
			# if hasil_predict[0]>1:
			# 	st.write('Hasil prediksi nilai efisiensi : 1')
			# else:
			# 	st.write(f'Hasil prediksi nilai Efisiensi : {hasil_predict[0]:.2f}')
			st.write(f'Hasil prediksi nilai Efisiensi : {round(hasil_predict[0],2):.2f}')
			if round(hasil_predict[0],2)>1.00:
				st.write('sudah mencapai efisien tetapi Variabel input masih bisa ditambah atau variabel output masih bisa dikurangi')
			elif round(hasil_predict[0],2)<1.00:
				st.write('Belum efisien variabel input masih bisa dikurangi atau variabel outuput masih bisa ditambah')
			else:
				st.write('sudah mencapai efisien')
	with col2:	
		if st.button("Clear"):
			st.write('clear')

	# with st.expander('Data Scale'):
	# 	#hasil=minmax_scaler(data_kolom,contoh)
	# 	data = {'kolom_1': x1.split(','),
	# 	        'kolom_2': x2.split(','),
	# 	        'kolom_3': x3.split(','),
	# 	        'kolom_4': x4.split(','),
	# 	        'kolom_5': x5.split(','),
	# 	        'kolom_6': x6.split(','),
	# 	        'kolom_7': x7.split(','),
	# 	        'kolom_8': x8.split(','),
	# 	        'kolom_9': x9.split(','),
	# 	        'kolom_10': x10.split(','),
	# 	        'kolom_11': x11.split(','),
	# 	        'kolom_12': x12.split(','),
	# 	        'kolom_13': x13.split(',')
	# 	        }


	# 	data_frame=pd.DataFrame(data, columns=['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN', 
	# 		'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI'])
	# 	st.dataframe(data_frame)

	# with st.expander('Hasil Prediksi'):
	# 	model=load('model/model_gb.joblib')
	# 	#hasil_predict=model.predict(data_frame)
	# 	#st.write(hasil_predict)