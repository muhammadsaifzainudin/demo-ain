import streamlit as st
from io import StringIO
import base64
import pandas as pd
from text_modelling import predict_community_topic as pred_top
from text_modelling import predict_sentiment as pred_sen
from text_modelling import text_preprocessor as tp
from stqdm import stqdm
from datetime import datetime
#import subprocess




stqdm.pandas()


#def run_cmd(args_list):
#  print('Running system command: {0}'.format(' '.join(args_list)))
#  proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#  s_output, s_err = proc.communicate()
#  s_return =  proc.returncode
  
#  return s_return, s_output, s_err

@st.cache(suppress_st_warning=True) 
def processing_ain(dataframe):
  sentiment_list = list()
  topic_list = list()
    
  for i in stqdm(dataframe.index):
    text = str(dataframe["valuesurvey.communityexperience.question.feedback"][i])
      
    if text == 'nan' or text == " ":
      sentiment_list.append("No Comment")
      topic_list.append("No Comment")
        
    else:
      processed_text = tp.text_preprocessor(text)
        
      if processed_text == '':
        sentiment_list.append("Unspecified Comments")
        topic_list.append("Unspecified Comments")
          
      else:
        sentiment_list.append(pred_sen.predict(processed_text)[1])
        topic_list.append(pred_top.predict_community_topic(processed_text)[1])
          
  dataframe["Attributes"] = topic_list
  dataframe["Sentiment"] = sentiment_list
  
  return dataframe
    
      
def save_json(df):
  file_download = f"unifi_community_processed_{str(datetime.now().strftime('%Y%m%d'))}.csv"
  #file_hdfs = f"text_modelling/dataset/unifi_community_processed_{str(datetime.now().strftime('%Y%m%d'))}.csv"

  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode() # some strings <-> bytes conversions necessary here
  href = f'<a href="data:file/csv;base64,{b64}" download="{file_download}" ><button>Download csv file</button></a>'
  
  #df.to_csv(file_hdfs, index = False )
  #HDFS_PATH = f"/user/TM36894/unifi_community_result/"   
  #cmd = ['hdfs', 'dfs', '-put', file_hdfs, HDFS_PATH]
  #ret, out, err = run_cmd(cmd)
  
  return href

def combine_data(hc, nps):
  file_download = f"combined_nps_{str(datetime.now().strftime('%Y%m%d'))}.csv"
  hc.columns = ['CSP Name', 'Name', 'XID',  'Team' ]
  hc['CSP Name'] = hc['CSP Name'].apply(lambda x: x.upper())
  nps['Staff Name'] = nps['Staff Name'].apply(lambda x: x.upper())

  output = nps.merge(hc[['CSP Name', 'XID','Team']], how = 'left', left_on = 'Staff Name', right_on=  'CSP Name')

  csv = output.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode() # some strings <-> bytes conversions necessary here
  href = f'<a href="data:file/csv;base64,{b64}" download="{file_download}" ><button>Download csv file</button></a>'

  return href
  

add_selectbox = st.sidebar.selectbox("Navigation", ("AIN: Unifi Community", "AIN: NPS"))

if add_selectbox == 'AIN: Unifi Community':
  st.title("Welcom to AIN: Unifi Community")  
  uploaded_file = st.file_uploader("Choose a file", type = ["csv"] )
  if uploaded_file is not None:
    dataframe =  pd.read_csv(uploaded_file)
  
    if not "valuesurvey.communityexperience.question.feedback" in dataframe.columns:
      st.write("The file does not contain customer feedback")
  
    else:
      df = processing_ain(dataframe)
      st.write(df)
      st.markdown(save_json(df), unsafe_allow_html=True)

else:
  st.title("Welcome to AIN: NPS") 
  st.write("1. Please upload HC file")
  hc_file_upload = st.file_uploader("Choose HC file", type = ["csv"] )
  if hc_file_upload is not None:
    hc = pd.read_csv(hc_file_upload)
    st.write("2. Please upload NPS file")
    nps_file_upload = st.file_uploader("Choose NPS file", type = ["csv"] )

    if nps_file_upload is not None:
      nps = pd.read_csv(nps_file_upload)
      st.write("3. Here is your combined file")
      st.markdown(combine_data(hc, nps), unsafe_allow_html=True )

    
      
      
    
  

    
    
