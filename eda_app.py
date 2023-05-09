import streamlit as st 
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px 


@st.cache_data
def load_data(data):
	df = pd.read_csv(data)
	return df

def normalize(data):
	scaler = MinMaxScaler()
	ndf = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
	return ndf


def run_eda_app():
	st.subheader("EDA Section")
	df=load_data('data/data_fix.csv')
	submenu = st.sidebar.selectbox("Submenu",['Data Awal','Normalisasi'])
	if submenu == 'Data Awal':

		
		with st.expander('Tabel'):
			st.dataframe(df)

		with st.expander("Tipe Data"):
		#	st.dataframe(df.dtypes)
			st.dataframe(df.dtypes)

		with st.expander("Ukuran Data"):
			st.dataframe(df.shape)

		with st.expander('Statistik Deskiptif'):
			st.dataframe(df.describe().transpose())

		with st.expander('Analisis Distribusi'):
			col1,col2 = st.columns([2,2])
			with col1:
				#with st.expander ('Dist Plot of Distibrution'):
					st.write('Distribusi WP BENDAHARA')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="WP_BENDAHARA", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Jumlah WP Badan')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="WP_BADAN", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi WP OP Pengusaha')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="WP_OP_PENGUSAHA", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Jumlah Account Representative')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="JUMLAH_AR", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Kepatuhan Penyampaian SPT')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="KEPATUHAN_SPT", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Pertumbuhan Penerimaan')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="PERTUMBUHAN_PENERIMAAN", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Jumlah SP2DK Cair')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="SP2DK_CAIR", kde=True, palette="deep")
					st.pyplot(fig)


			with col2:
				#with st.expander ('Dist Plot of Distibrution'):
					st.write('Distribusi WP OP Karyawan')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="WP_OP_KARYAWAN", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Jumlah Fungsional Pemeriksa')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="JUMLAH_FPP", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Realisasi Anggaran')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="REALISASI_ANGGARAN", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Capaian Penerimaan Pajak')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="CAPAIAN_PENERIMAAN", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Jumlah SP2DK Terbit')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="SP2DK_TERBIT", kde=True, palette="deep")
					st.pyplot(fig)

					st.write('Distribusi Jumlah Pemeriksaan Selesai')
					fig, ax = plt.subplots(figsize=(5,3))
					sns.histplot(data=df, x="PEMERIKSAAN_SELESAI", kde=True, palette="deep")
					st.pyplot(fig)


		with st.expander("Sebaran"):
			data_kolom=df[['WP_BENDAHARA', 'WP_BADAN','WP_OP_KARYAWAN', 'WP_OP_PENGUSAHA', 'JUMLAH_FPP', 'JUMLAH_AR','REALISASI_ANGGARAN', 
			'KEPATUHAN_SPT', 'CAPAIAN_PENERIMAAN','PERTUMBUHAN_PENERIMAAN', 'SP2DK_TERBIT', 'SP2DK_CAIR','PEMERIKSAAN_SELESAI']]
			pca_model = PCA(n_components=2)
			pca_data = pca_model.fit_transform(data_kolom)
			df_pca = pd.DataFrame(pca_data, columns=['A', 'B'])
			fig, ax = plt.subplots(figsize=(8,6))
			sns.scatterplot(data=df_pca, x="A", y="B",palette="deep")
			st.pyplot(fig)

		with st.expander("Outlier"):

			fig, ax = plt.subplots()
			sns.boxplot(data=df, orient='h', ax=ax)
			ax.set_title('Boxplot')
			#ax.set_xlabel('Keterangan Sumbu X')
			#ax.set_ylabel('Keterangan Sumbu Y')
			st.pyplot(fig)

			df_norm = normalize(data_kolom)
			fig, ax = plt.subplots()
			sns.boxplot(data=df_norm, orient='h', ax=ax)
			ax.set_title('Boxplot Normalisasi')
			#ax.set_xlabel('Keterangan Sumbu X')
			#ax.set_ylabel('Keterangan Sumbu Y')
			st.pyplot(fig)

		with st.expander("Korelasi"):

			
			corr = df.corr()

			#fig, ax = plt.subplots()
			#sns.scatterplot(x=data['kolom_1'], y=data['kolom_2'], ax=ax)
			#ax.set_title('Korelasi antara kolom_1 dan kolom_2')

			#st.pyplot(fig)

			fig, ax = plt.subplots()
			sns.heatmap(corr, cmap='coolwarm', annot=True, ax=ax, annot_kws={"size":5})
			ax.set_title('Matriks Korelasi')
			plt.xticks(rotation=80)
			st.pyplot(fig)
			
			


	else:

		st.subheader('Plots')

		col1,col2 = st.columns([2,1])
		with col1:
			with st.expander ('Dist Plot of Distibrution'):
				st.write('Dist')