import streamlit as st 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


@st.cache_data
def load_data(data):
	df = pd.read_csv(data)
	return df


def pilih_kolom(data):
	data_kolom=data[['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN', 
			'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI']]
	return data_kolom

def variabel_input(data):
	v_in= data[['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN']]
	return v_in

def variabel_output(data):
	v_out= data[[ 'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI']]
	return v_out


def minmax_scaler(data):
	scaler = MinMaxScaler()
	data_mm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
	return data_mm

def standar_scaler(data):
	scaler = StandardScaler()
	df_ss= pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
	return df_ss

def zcsore_scalser(data):
	df_zs=pd.DataFrame(data.apply(zscore), columns=data.columns)
	return df_zs

def transform_scaler(data):
	normalized_df = (data - data.min()) / (data.max() - data.min())
	log_normalized_df = np.log(normalized_df)
	return log_normalized_df


def iqr_outlier_detection(df):
    # tentukan batas bawah dan atas untuk setiap kolom menggunakan metode IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # temukan outlier menggunakan fungsi loc pada setiap kolom
    outliers = pd.DataFrame()
    for col in df.columns:
        col_outliers = df.loc[(df[col] < lower_bound[col]) | (df[col] > upper_bound[col])]
        col_outliers['column'] = col
        outliers = pd.concat([outliers, col_outliers])

    # kembalikan DataFrame outlier
    return outliers


def run_prep_app():
	st.subheader("Data Preparation")

	#df=load_data('data/data_fix.csv')

	data_=st.file_uploader("Upload Data Set", type=['csv'])
	submenu = st.sidebar.selectbox("Submenu",['Data Awal','Normalisasi'])
	if data_ is not None:
		data_awal=pd.read_csv(data_)
		

		#data_awal = load_data('data/data_fix.csv')
		data_kolom = pilih_kolom(data_awal)
		submenu = st.sidebar.selectbox("Submenu",['Normalisasi','Pemilihan Variabel','Deteksi Outlier'])
		

		if submenu == 'Normalisasi':
			st.write('Normalisasi')
			

			with st.expander('Min Max Scaler'):
				hasil_minmax=minmax_scaler(data_kolom)
				st.dataframe(hasil_minmax)

			with st.expander('Standard Scaler'):
				hasil_ss=standar_scaler(data_kolom)
				st.dataframe(hasil_ss)

			with st.expander('ZScore Scaler'):
				hasil_zs=zcsore_scalser(data_kolom)
				st.dataframe(hasil_zs)

			with st.expander('Transform Log Scaler'):
				hasil_ts= transform_scaler(data_kolom)
				st.dataframe(hasil_ts)


			with st.expander('Hasil'):

				col1,col2 = st.columns([2,2])
				with col1 : 
					st.write('Min Max Scaler')
					min_mm= hasil_minmax.min()
					max_mm= hasil_minmax.max()
					min_max = pd.concat([min_mm, max_mm], axis=1)
					min_max.columns=['Min','Max']
					st.dataframe(min_max)

					st.write('ZScore Scaler')
					min_zs= hasil_zs.min()
					max_zs= hasil_zs.max()
					min_max_zs = pd.concat([min_zs, max_zs], axis=1)
					min_max_zs.columns=['Min','Max']
					st.dataframe(min_max_zs)


				with col2:
					st.write('Standard Scaler')
					min_ss= hasil_ss.min()
					max_ss= hasil_ss.max()
					min_max_ss = pd.concat([min_ss, max_ss], axis=1)
					min_max_ss.columns=['Min','Max']
					st.dataframe(min_max_ss)

					st.write('Transform Log Scaler')
					hts= transform_scaler(data_kolom)
					min_ts= hts.min().astype(str)
					max_ts= hts.max()
					min_max_ts = pd.concat([min_ts, max_ts], axis=1)
					min_max_ts.columns=['Min','Max']
					#st.write(hts['WP_BENDAHARA'].min())
					st.dataframe(min_max_ts)

		if submenu=='Pemilihan Variabel':


			st.write('Pemilihan Variabel')

			v_in_out = minmax_scaler(data_kolom)

			with st.expander('Variabel Input'):
				st.write('Variabel Input')
				vr_in = variabel_input(v_in_out)
				vi = pd.concat([data_awal['KD_KPP'],vr_in], axis=1)
				st.dataframe(vi)



			with st.expander('Variabel Output'):
				st.write('Variabel Output')
				vr_out= variabel_output(v_in_out)
				vo = pd.concat([data_awal['KD_KPP'],vr_out], axis=1)
				st.dataframe(vo)

		if submenu=='Deteksi Outlier':
			st.write('Deteksi Outlier')
			v_in_out = minmax_scaler(data_kolom)
			vr_in = variabel_input(v_in_out)
			with st.expander('Boxplot'):
				fig, ax = plt.subplots()
				sns.boxplot(data=vr_in, orient='h', ax=ax)
				ax.set_title('Boxplot Outlier')
				#ax.set_xlabel('Keterangan Sumbu X')
				#ax.set_ylabel('Keterangan Sumbu Y')
				st.pyplot(fig)
			with st.expander('DataFrame'):
				outliers = iqr_outlier_detection(vr_in)
				if outliers.empty:
					st.write('Tidak ada Outlier')
				else:
					st.write('Outlier :')
					st.write(outliers)
			with st.expander('Hasil'):		
					bend=outliers[outliers['column']=='WP_BENDAHARA']['WP_BENDAHARA'].count()
					bdn=outliers[outliers['column']=='WP_BADAN']['WP_BADAN'].count()
					opk=outliers[outliers['column']=='WP_OP_KARYAWAN']['WP_OP_KARYAWAN'].count()
					opp=outliers[outliers['column']=='WP_OP_PENGUSAHA']['WP_OP_PENGUSAHA'].count()
					fpp=outliers[outliers['column']=='JUMLAH_FPP']['JUMLAH_FPP'].count()
					ar=outliers[outliers['column']=='JUMLAH_AR']['JUMLAH_AR'].count()
					angg=outliers[outliers['column']=='JUMLAH_AR']['JUMLAH_AR'].count()
					data_outliers=[['WP_BENDAHARA',bend],['WP_BADAN',bdn],['WP_OP_KARYAWAN',opk],['WP_OP_PENGUSAHA',opp],
					['JUMLAH_FPP',fpp],['JUMLAH_AR',ar],['REALISASI_ANGGARAN',angg]]
					hasil=pd.DataFrame(data_outliers,columns=['Variabel','Jumlah Outlier'])
					st.dataframe(hasil)
				
	else:
		st.write('data tidak ada')