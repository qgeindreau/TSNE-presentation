from django.conf.urls import url

from . import views

app_name = 'MesAlgo'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^preprocessing/$', views.prepro, name='prepro'),
    url(r'^ClusteringSpectral/$', views.clustering, name='clustering'),
    url(r'^Lien/$', views.lien, name='lien'),
    url(r'^MiseEnOeuvre/$', views.meo, name='meo'),       
]
