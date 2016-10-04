from etsy import app

from flask import Flask,render_template,request,redirect
import pandas as pd
import numpy as np
import pickle
from random import randint
from scipy.special import erf

#import requests
import os
import sys
import time
import operator


from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource
from bokeh.charts import Area
from bokeh.models import Range1d
from bokeh.models import HoverTool, Label
from bokeh.io import output_file, vplot
from bokeh.models import ColumnDataSource, Range1d, Plot, LinearAxis, Grid
from bokeh.models.glyphs import ImageURL



#from bokeh.plotting import figure, show, output_file

#import scipy.special


###############################################################################
## Python functions - loading
###############################################################################

def load_testids():  
  NBOW, random_state = 1000, 30
  with open('tmp/OBJ/model/splits_{}_{}.pkl'.format(random_state, NBOW)) as f:
    listing_ids, train_ids, test_ids = pickle.load(f)
  return test_ids

def load_df():
  NBOW, random_state = 1000, 30
  print ('Loading data...')    
  start = time.time()
  df=pd.read_csv('tmp/OBJ/meta/df_final.csv', encoding='utf-8')
  df.drop(['Unnamed: 0'],axis=1, inplace=True)
  print '{:2.2f} seconds'.format(time.time() - start) 
  return df

def extract_raw_data(df,id) :
  dft = df[df['listing_id']==float(id)]        
  price=dft['price'].astype(float).tolist()[0]
  urlMed=dft['urlMed'].tolist()[0]
  urlLarge=dft['urlLarge'].tolist()[0]
  out, title, description, materials, tags, styles = compile_text(dft)
  return price, title, description, materials, tags, styles, out, urlMed, urlLarge

def extract_data(df, id, out):
  dft = df[df['listing_id']==float(id)]        
  bow = load_bow()
  Xt = bow.transform([out]).toarray()    
  Xn = dft.iloc[:,2:11].astype(int).values.tolist()
  Xi = dft.iloc[:,16:52].astype(float).values.tolist()        
  return Xt, Xn, Xi

def compile_text(dft):
  title = dft['title'].tolist()[0]
  desc = dft['description'].tolist()[0]
  mat = dft['materials'].tolist()[0]
  tag = dft['tags'].tolist()[0]
  style = dft['style'].tolist()[0]
  if style == None:
      style = 'a'
  elif isinstance(style, float):
      style='a'
  if isinstance(title, float):
      title='a'
  if isinstance(tag, float):
      tag='a'
  if isinstance(mat, float):
      mat='a'    
  tagOut=[ii for ii in tag.strip('[]').split(',')]
  titleOut=[ii for ii in title.strip('[]').split(',')]
  styleOut=[ii for ii in style.strip('[]').split(',')]
  descOut=[ii for ii in desc.strip('[]').split(',')]
  matOut=[ii for ii in mat.strip('[]').split(',')]      
  Out = tagOut + titleOut + styleOut + descOut + matOut
  out = list(([ out for out in Out]))
  out = " ".join( out )    
  return out, title, desc, mat, tag, style

def load_bow():
  NBOW, random_state = 1000, 30
  with open('tmp/OBJ/model/bow_{}.pkl'.format(NBOW)) as f:
    bow = pickle.load(f)   
  return bow

def load_models():
  NBOW, random_state = 1000, 30
  print "Loading models..."
  start = time.time()    
  with open('tmp/OBJ/model/forest1_{}.pkl'.format(NBOW)) as f:
    forest_r1 = pickle.load(f)
  with open('tmp/OBJ/model/forest2_{}.pkl'.format(NBOW)) as f:
    forest_r2 = pickle.load(f)
  print '{:2.2f} seconds'.format(time.time() - start)
  return forest_r1, forest_r2

###############################################################################
## Python functions - demo
###############################################################################  

def gen_demoListings():
  test_ids = load_testids()
  df = load_df() 
  demo_ids=[]
  demo_urls=[]

  ## fill list with randomly pulled samples from testing set
  for i in range(9):    
    demo_id = int(test_ids[  randint(0, len(test_ids))  ])
    price, title, description, materials, tags, styles, out, urlMed, urlLarge = extract_raw_data(df,demo_id)
    demo_ids.append(demo_id)
    demo_urls.append(urlLarge)

  ## to ensure something close to the norm for my presentation, prepare a few curated selections
  curated_ids = [124386444, 229795748, 256569756, 246713718, 463098088, 462750562, 241235302, 66770733]
  demo_id = int(curated_ids[  randint(0, len(curated_ids))  ])
  price, title, description, materials, tags, styles, out, urlMed, urlLarge = extract_raw_data(df,demo_id)
  demo_ids[4]=demo_id
  demo_urls[4]=urlLarge

  return demo_ids, demo_urls

###############################################################################
## Python functions - query
###############################################################################

def query(listing_id):  
  df = load_df() 
  price, title, description, materials, tags, styles, out, urlMed, urlLarge = extract_raw_data(df,listing_id)
  meta = [description]+[materials]+[price]+[styles]+[tags]+[title]+[urlMed]+[urlLarge]

  Y = predict(df, listing_id, out)
  
  arr = calc_similarity(df, listing_id)
  bot_ids, bot_sims, bot_urls, bot_prices, top_ids, top_sims, top_urls, top_prices = get_hits(df = df, arr=arr, Y = Y, N = 6)  
  
  
  d1 = dict(zip(top_ids, top_prices))
  d2 = dict(zip(top_urls, top_prices))
  d3 = dict(zip(top_sims, top_prices))
  sorted_ids_temp = sorted(d1.items(), key=operator.itemgetter(1))
  sorted_urls_temp = sorted(d2.items(), key=operator.itemgetter(1))
  sorted_sims_temp = sorted(d3.items(), key=operator.itemgetter(1))
  
  sorted_prices = sorted(top_prices)  
  sorted_ids = [s[0] for s in sorted_ids_temp]
  sorted_urls = [s[0] for s in sorted_urls_temp]
  sorted_sims = [s[0] for s in sorted_sims_temp]

  
  #return top_ids, top_sims, top_urls, top_prices, meta, Y
  return sorted_ids, sorted_sims, sorted_urls, sorted_prices, meta, Y
          

def calc_similarity(df, id):
  # load reference
  dft = df[df['listing_id']==float(id)]          
  y_ref = dft.iloc[:,16:52].astype(float).values.tolist()[0]
  # load database
  listing_ids=df['listing_id'].astype(int).astype(str).tolist()
  Xi = df.iloc[:,16:52].astype(float).values.tolist()
  # calc similarity
  i=0
  arr = []
  for x, listing_id in zip(Xi, listing_ids):    
    ## normalize all vectors
    y =  x/np.sqrt(abs(np.dot( x, x)))
    ## calculate similarity
    sim  = np.dot(y,y_ref)    
    ## build list for later sorting
    arr.append((i, sim, listing_id))    
    ## increment
    i+=1  
  return arr


def get_hits(df, arr, Y, N):
  urlMed=df['urlMed'].tolist()
  price=df['price'].astype(float).tolist()
  top = sorted(arr, key=lambda k: k[1], reverse = True)[1:(10*N+1)]
  top_ids, top_sims, top_urls, top_prices = [], [], [], []
  for t in top:
    i, sim, style= t
    if abs(Y-price[i])<5*0.3*Y: # only take values that fall in range
      top_sims.append(str(sim))
      top_ids.append(str(int(style)))
      top_urls.append(urlMed[i])
      top_prices.append(price[i])
    if len(top_ids)==6:
      break
  bot = sorted(arr, key=lambda k: k[1], reverse = False)[1:(N+1)]
  bot_ids, bot_sims, bot_urls, bot_prices = [], [], [], []
  for b in bot:
    i, sim, style = b
    bot_sims.append(str(sim))
    bot_ids.append(str(int(style)))
    bot_urls.append(urlMed[i])
    bot_prices.append(price[i])
  return bot_ids, bot_sims, bot_urls, bot_prices, top_ids, top_sims, top_urls, top_prices


def predict(df, id, out):

  Xt, Xn, Xi = extract_data(df, id, out)
  forest_r1, forest_r2 = load_models()

  ## make first prediction
  Y1 = forest_r1.predict(Xt)[0]
  ## prepare for second prediction        
  X2  = np.concatenate((Xn, Xi, Y1[None,None]),axis=1)
  # Predict
  Y2 = np.exp(forest_r2.predict(X2[0])[0])

  return Y2

###############################################################################
## Python functions - Bokeh
###############################################################################

def generateBokeh1(Y, top_ids, top_prices, price, listing_id): 
  print ('Generating Bokeh ...')
  p = figure(plot_height=220, plot_width=1000, background_fill_color="#ffffff",x_axis_location="above",)
  
  mu, sigma = Y, 0.3*Y  
  y_price = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(price-mu)**2 / (2*sigma**2))

  L, N = 5*sigma, 50.  
  dL=2.*L/N
  h=1.2/(sigma * np.sqrt(2*np.pi)) 

  color2,x2,pdf2 = [],[],[]
  for i in range(int(N)):
    
    #c1 = 255 * np.exp(-(i-N/2)**2 / (2*(L/sigma)**2))
    c1 = 255*(N-i)/N
    c2 = 255-c1
    color=(c2, 125, c1)  

    x = np.linspace(-L+(i)*dL, -L+(i+1)*dL, 200)+mu
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))

    x = np.concatenate((x,[x[-1]],[x[0]]))
    pdf = np.concatenate((pdf,[0],[0]))
    
    color2.append('#%02x%02x%02x' % color)
    x2.append(x)
    pdf2.append(pdf)

  p.patches(xs=x2, ys=pdf2, color=color2, alpha=0.6, line_alpha=0)
  p.set(x_range=Range1d(mu-5*sigma, mu+5*sigma), y_range=Range1d(0, h))  
  p.xaxis.axis_label = 'Price'
  
  for top_price in top_prices:            
    top_price_y = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(top_price-mu)**2 / (2*sigma**2))
    p.circle(x=top_price, y=top_price_y, size=20,color='green', alpha=0.5, line_alpha=0.5)      
    p.line(x=[top_price,top_price], y=[0,top_price_y], color='green', alpha=0.5, line_width=5)        
  p.triangle(x=price, y=y_price, size=20,color='red', line_alpha=0.8)        
  p.inverted_triangle(x=price, y=y_price, size=20,color='red', line_alpha=0.8)        
  p.yaxis.visible = False  
  
  p.line(x=[price,price], y=[0,1], color='#F05F40', line_alpha=0.5, line_width=5)        

  p.triangle(x=Y, y=0.94*h, size=20,color='#212121', alpha=1)        
  p.line(x=[Y,Y], y=[0,0.94], color='#212121', line_alpha=1, line_width=3)       

  p.border_fill_color = None
  p.logo = None
  p.toolbar_location = None
  p.ygrid.grid_line_color = None
  p.xaxis.axis_label_text_font_size = "16pt"


  rge=[mu-5*sigma, mu+5*sigma]
  drge=(rge[1]-rge[0])/(6-1.)

  alpha = 0.50+0.35*erf((price-Y)/sigma)
  print alpha, 1-alpha
  mytext1 = Label(x=mu+2.5*sigma, y=0.8*h, text='Above market value', text_font_size='20pt', text_alpha=alpha)
  mytext2 = Label(x=mu-4.5*sigma, y=0.8*h, text='Below market value', text_font_size='20pt', text_alpha=1-alpha)

  p.add_layout(mytext1)
  p.add_layout(mytext2)




  p2 = figure(plot_height=80, plot_width=1000, background_fill_color="#ffffff")
  color2,x2,pdf2 = [],[],[]
  for i in range(int(N)):
    
    # c1 = 255 * np.exp(-(i-N/2)**2 / (2*(L/sigma)**2))    
    # c2 = 255-c1
    # color=(c1, 0, c2)  
    c1 = 255*(N-i)/N
    c2 = 255-c1
    color=(c2, 125, c1)  

    x = np.linspace(-L+(i)*dL, -L+(i+1)*dL, 200)+mu
    pdf = x*0 +1

    x = np.concatenate((x,[x[-1]],[x[0]]))
    pdf = np.concatenate((pdf,[0],[0]))
    
    color2.append('#%02x%02x%02x' % color)
    x2.append(x)
    pdf2.append(pdf)
 
  p2.patches(xs=x2, ys=pdf2, color=color2, alpha=0.6, line_alpha=0)
  p2.set(x_range=Range1d(mu-5*sigma, mu+5*sigma), y_range=Range1d(0, 1))  
  p2.xaxis.axis_label = 'Price'
  p2.yaxis.visible = False    
  p2.xaxis.visible = False
  p2.ygrid.grid_line_color = None

  for ipos, top_price in zip(np.arange(rge[0],rge[1]+drge,drge),top_prices):      
    top_price_y = 1
    p2.line(x=[top_price, top_price], y=[0.3,1], color='#266d6d', line_alpha=0.5, line_width=4)      
    p2.line(x=[ipos, top_price], y=[0,0.3], color='#266d6d', line_alpha=0.5, line_width=4)      
    #p2.circle(x=ipos, y=0, size=20,color='green', alpha=0.5, line_alpha=0.5)      
  #p2.line(x=[price,price], y=[0,1], color='red', line_alpha=0.5, line_width=8)        
  #p2.line(x=[Y,Y], y=[0,1], color='black', line_alpha=0.5, line_width=8)        
  
  p2.border_fill_color = None
  p2.logo = None
  p2.toolbar_location = None 

  pf = vplot(p,p2)
  script, div = components(pf)
  return script, div

###############################################################################
## Flask redirects
###############################################################################

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index',methods=['GET','POST'])
def index():
  sys.stdout.flush()
  if request.method == 'GET':    
    demo_ids, demo_urls = gen_demoListings()
  return render_template('index.html', listing_id = demo_ids[0], demo=zip(demo_ids,demo_urls), n=3)   
  
@app.route('/try_app',methods=['GET','POST'])
def try_app():
  sys.stdout.flush()  
  if request.method == 'POST':        
    listing_id = request.form['listing_id']  
    top_ids, top_sims, top_urls, top_prices, meta, Y = query(listing_id)

    #meta = [description]+[materials]+[price]+[styles]+[tags]+[title]+[urlMed]+[urlLarge]
    desc = meta[0]
    material = meta[1]
    price = meta[2]
    style = meta[3]
    tags = meta[4]
    title=meta[5]
    urlMed = meta[6]
    urlLarge = meta[7]
    
    script, div = generateBokeh1(Y, top_ids, top_prices, price, listing_id)      

    return render_template('try_app.html', listing_id = listing_id, tops = zip(top_ids,top_urls), price=price,title=title,desc=desc, material=material, style=style, tags=tags, Y=Y, imageUrl=urlLarge, script=script, div=div)   

