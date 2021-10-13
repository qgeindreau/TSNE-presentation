from re import template
from django.shortcuts import render, get_object_or_404
import pandas as pd
import numpy as np
from .models import Carousel
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.decomposition import PCA 
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from .forms import FigForm
def index(request):
    context = {'articles': Carousel.objects.all(),}
    return render(request, "MesAlgo/index.html", context)

def prepro(request):
    Visus=[]
    data_sets=['contact-lenses','pasture','squash-stored','newthyroid','car','bondrate']
    dfs=[pd.read_csv('csv/'+i+'.csv',header=None) for i in data_sets]
    for i in range(len(dfs)):
        dfs[i].columns=list(dfs[i].columns)[:-1]+['classe']
    index=0
    for df in dfs:
        Visus.append(data_sets[index])
        if len(df.columns)<=20:
            fig=ff.create_table(df.iloc[:5], height_constant=20)
            Visus.append(fig.to_html(full_html=False, default_height=500, default_width=1000))
        else:
            fig=ff.create_table(df.iloc[:5,:20], height_constant=20)
            Visus.append(fig.to_html(full_html=False, default_height=500, default_width=1000))
            fig=ff.create_table(df.iloc[:5,20:], height_constant=20)
            Visus.append(fig.to_html(full_html=False, default_height=500, default_width=1000))
        Visus.append(df.shape)
        index+=1
    context={'visus':Visus}
    return render(request, 'MesAlgo/kmeans.html', context)

def clustering(request):
    Visus=[]
    data_sets=['contact-lenses','pasture','squash-stored','newthyroid','car','bondrate']
    dfs=[pd.read_csv('csv/'+i+'.csv',header=None) for i in data_sets]
    for i in range(len(dfs)):
        dfs[i].columns=list(dfs[i].columns)[:-1]+['classe']    
    class Visionneur:
        def __init__(self,nom,visu):
            self.nom=nom
            self.acp=visu[0]
            self.tsne=visu[1]
    for df in dfs:
        pca=PCA()
        perp=np.sqrt(len(list(df.columns)))
        tsne1=TSNE(n_components=1,n_jobs=-1,perplexity=perp,n_iter=4000)
        tsne2=TSNE(n_components=2,n_jobs=-1,perplexity=perp,n_iter=4000)
        tsne3=TSNE(n_components=3,n_jobs=-1,perplexity=perp,n_iter=4000)
        features=list(df.columns[:-1])
        components = pca.fit_transform(StandardScaler().fit_transform(df[features]))
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.4, 0.6],
            subplot_titles=("ACP","Histogramme variance"),
            specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter","rowspan": 2}],
                [            None                 , None       ]])
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=components[:,0],
                y=components[:,1],
                marker_color=df['classe']
                ),
                row=1,col=1)
        fig.add_trace(
            go.Bar(
                x=[i for i in range(1,len(pca.explained_variance_ratio_)+1)],
                y=[x*100 for x in pca.explained_variance_ratio_]),
                row=1,col=2)
        fig.update_layout(template="plotly_dark")
        ########################################################################################
        projections1 = tsne1.fit_transform(StandardScaler().fit_transform(df[features]), )
        projections2 = tsne2.fit_transform(StandardScaler().fit_transform(df[features]), )
        projections3 = tsne3.fit_transform(StandardScaler().fit_transform(df[features]) )
        fig2=make_subplots(
            rows=2, cols=2,
            column_widths=[0.6, 0.4],
            row_heights=[0.4, 0.6],
            subplot_titles=("2D","1D","3D"),
            specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                [            None                 , {"type": "scatter3d"}]])
        fig2.add_trace(go.Scatter(
            mode='markers',
            x=projections2[:,0],
            y=projections2[:,1],
            marker_color=df['classe']),
            row=1,col=1)
        fig2.add_trace(go.Scatter(
            mode='markers',
            x=projections1[:,0],
            y=projections1[:,0],
            marker_color=df['classe']),
            row=1,col=2
            )
        fig2.add_trace(go.Scatter3d(
            mode='markers',
            x=projections3[:,0],
            y=projections3[:,1],
            z=projections3[:,2],
            marker_color=df['classe'],
            marker_size=5),
            row=2,col=2)
        Visus.append((fig.to_html(full_html=False, default_height=500, default_width=1000),fig2.to_html(full_html=False, default_height=500, default_width=1000)))
    rendu=[]
    index=0
    for figs in Visus:
        rendu.append(Visionneur(data_sets[index],figs))
        index+=1
    context={'visus':rendu
        }
    return render(request, 'MesAlgo/clustering.html', context)







def lien(request):
    if request.method == 'POST':
        Fig_form = FigForm(request.POST, request.FILES)
        if Fig_form.is_valid:
            perp=float(request.POST['perp'])
            lrate=int(request.POST['l_rate'])
            nbiter=int(request.POST['nbiter'])
            df_name=request.POST['csv']
            df=pd.read_csv('csv/'+df_name+'.csv',header=None)
            df.columns=list(df.columns)[:-1]+['classe']    
            tsne1=TSNE(n_components=1,n_jobs=-1,perplexity=perp,n_iter=nbiter,learning_rate=lrate,init='pca')
            tsne2=TSNE(n_components=2,n_jobs=-1,perplexity=perp,n_iter=nbiter,learning_rate=lrate,init='pca')
            tsne3=TSNE(n_components=3,n_jobs=-1,perplexity=perp,n_iter=nbiter,learning_rate=lrate,init='pca')
            features=list(df.columns[:-1])
            projections1 = tsne1.fit_transform(StandardScaler().fit_transform(df[features]) )
            projections2 = tsne2.fit_transform(StandardScaler().fit_transform(df[features]) )
            projections3 = tsne3.fit_transform(StandardScaler().fit_transform(df[features]) )
            fig2=make_subplots(
                rows=2, cols=2,
                column_widths=[0.6, 0.4],
                row_heights=[0.4, 0.6],
                subplot_titles=("2D","1D","3D"),
                specs=[[{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                    [            None                 , {"type": "scatter3d"}]])
            fig2.add_trace(go.Scatter(
                mode='markers',
                x=projections2[:,0],
                y=projections2[:,1],
                marker_color=df['classe']),
                row=1,col=1)
            fig2.add_trace(go.Scatter(
                mode='markers',
                x=projections1[:,0],
                y=projections1[:,0],
                marker_color=df['classe']),
                row=1,col=2
                )
            fig2.add_trace(go.Scatter3d(
                mode='markers',
                x=projections3[:,0],
                y=projections3[:,1],
                z=projections3[:,2],
                marker_color=df['classe'],
                marker_size=5),
                row=2,col=2)
            fig2.update_layout(template="plotly_dark")
            fig2.update_layout(height=600, width=800, title_text=df_name + '<br>Taille:'+str(df.shape))
            context={ 'form':FigForm(request.POST, request.FILES),
                    'test':fig2.to_html(full_html=False, default_height=550, default_width=1100),
                    }
        else:
            context={'form':FigForm()}
    else:
        context={'form':FigForm()}
    return render(request, 'MesAlgo/lien.html', context)
    
def meo(request):
    context={'form':FigForm()}
    return render(request, 'MesAlgo/meo.html', context)
