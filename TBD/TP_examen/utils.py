import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import progressbar

class Model(object):
    def __init__(self,classes):
        super(Model, self).__init__()
        self.classes = classes
        
    def learning(self,app_x,app_y,app_labels):
        pass
    
    def prediction(self,x,y):
        preds = self.classes.copy()
        shuffle(preds)
        return preds
    
    def test(self,dec_x,dec_y,dec_labels,print_results = True):
        dec_pred = [self.prediction(x,y) for x,y in zip(dec_x,dec_y)]
        dec_labels = list(dec_labels)
        top1,top2,CM = self.metrics(dec_labels,dec_pred)
        if print_results:
            print("Top 1 : ",top1)
            print("Top 2 : ",top2)
            print("Matrice de confusion : \n",str(CM).replace('[',' ').replace(']',''))
        return top1,top2,CM
    
    def metrics(self,gt,pred):
        CM = np.zeros((len(self.classes),len(self.classes)),dtype=np.int)
        top1 = 0
        top2 = 0
        for g,p in zip(gt,pred):
            if g == p[0]:
                top1+=1
                top2+=1
            elif g in p[:2]:
                top2+=1
            CM[np.where(self.classes==g)[0],np.where(self.classes==p[0])[0]]+=1
        return top1/len(gt), top2/len(gt), CM
    
    
def split_data(data,label_col):
    labels = list(np.unique(data[label_col]))
    return labels, [data[data[label_col]==l] for l in labels]


def plot_2_var(labels,all_x,all_y,titre=None,var1=None,var2=None):
    # Ici, labels, all_x et all_y sont des listes de même taille
    plt.figure(figsize=(6,6))
    for label,x,y in zip(labels,all_x,all_y):
        plt.plot(x,y,'o',label=label)
    plt.legend()
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(titre)
    plt.show()

    
def read_data(path,column_names=None,sep=' '):
    if column_names is not None:
        return pd.read_csv(path,header=None, sep=sep, names=column_names)
    else:
        return pd.read_csv(path,header=None, sep=sep)
########################################################################
#     CLASSES CLASSIFIEURS
########################################################################

class ClassifieurDistMin(Model):
    # Définition du constructeur
    def __init__(self,classes):
        super(ClassifieurDistMin, self).__init__(classes)
        self.classes = classes
        self.means = {c:None for c in classes}
    
    # Définition de la fonction d'apprentissage
    def learning(self,app_x,app_y,app_labels):
        for classe in self.classes:
            ind_lab = app_labels.index[app_labels==classe]
            app_x_lab = app_x[ind_lab]
            app_y_lab = app_y[ind_lab]
            self.means[classe] = np.array([np.mean(app_x_lab),np.mean(app_y_lab)])
    
    # Définition de la classe de calcul de distance
    def dist_classe(self,X,classe):
        M = self.means[classe]
        return (X-M).T@(X-M)
    
    # Définition de la classe de prédiction
    def prediction(self,x,y):
        X = np.array([x,y])
        distances = [self.dist_classe(X,classe) for classe in self.classes]
        indices = np.argsort(distances)
        return self.classes[indices]
    
class ClassifieurDistMahalanobis(Model):
    def __init__(self,classes):
        super(ClassifieurDistMahalanobis, self).__init__(classes)
        self.classes = classes
        self.means = {c:None for c in classes}
        self.covMat = {c:None for c in classes}
    def learning(self,app_x,app_y,app_labels):
        for classe in self.classes:
            ind_lab = app_labels.index[app_labels==classe]
            app_x_lab = app_x[ind_lab]
            app_y_lab = app_y[ind_lab]
            self.means[classe] = np.array([np.mean(app_x_lab),np.mean(app_y_lab)])
            self.covMat[classe] = np.linalg.inv(np.cov(app_x_lab,app_y_lab))
    
    def dist_classe(self,X,classe):
        M = self.means[classe]
        Z = self.covMat[classe]
        return (X-M).T@Z@(X-M)
    
    def prediction(self,x,y):
        X = np.array([x,y])
        distances = [self.dist_classe(X,classe) for classe in self.classes]
        indices = np.argsort(distances)
        return self.classes[indices]

def noyau_uniforme(A,B,h):
    # A et B sont 2 points 2D (x,y)
    # Noyau de taille h X h centré au point A
    # Si B est dans le noyau, la distance est de 1/h², sinon la distance est de 0
    if np.abs(A[0]-B[0])<h/2 and np.abs(A[1]-B[1])<h/2:
        return 1/h**2
    else:
        return 0

def noyau_gaussien(A,B,h):
    # A et B sont 2 points 2D (x,y)
    # Noyau gaussien de variance h centré au point A
    sigma = h
    return np.exp(-((A-B).T@(A-B))/(2*(sigma**2)))

class Parzen(Model):
    # Définition du constructeur
    def __init__(self,classes,noyau='uniforme',h=None):
        super(Parzen, self).__init__(classes)
        self.h = h
        assert noyau in ['uniforme','gaussien']
        self.noyau = noyau
        if self.noyau == 'uniforme':
            self.core_func = noyau_uniforme
        else:
            self.core_func = noyau_gaussien
    
    # Définition de la fonction d'apprentissage
    def learning(self,app_x,app_y,app_labels):
        self.app_points = {}
        for classe in self.classes:
            ind_lab = app_labels.index[app_labels==classe]
            app_x_lab = app_x[ind_lab]
            app_y_lab = app_y[ind_lab]
            self.app_points[classe]=[app_x_lab,app_y_lab]
    
    
    # Définition de la fonction de prédiction
    def prediction(self,x,y):
        X = np.array([x,y])
        classes_dists = np.zeros((len(self.classes)))
        for i,classe in enumerate(self.classes):
            classes_dists[i] = sum([self.core_func(X,pt,self.h) for pt in zip(self.app_points[classe][0],self.app_points[classe][1])])
        if np.all(classes_dists==0):
            return None
        ind_dists = np.argsort(classes_dists)[::-1]
        return self.classes[ind_dists]
    
    # Définition de la fonction de métrique qui prend en charge les points non classés.
    def metrics(self,gt,pred):
        CM = np.zeros((len(self.classes),len(self.classes)+1),dtype=np.uint16)
        top1 = 0
        top2 = 0
        for g,p in zip(gt,pred):
            if p is None:
                CM[np.where(self.classes==g)[0],-1]+=1
            else:
                if g == p[0]:
                    top1+=1
                    top2+=1
                elif g in p[:2]:
                    top2+=1
                CM[np.where(self.classes==g)[0],np.where(self.classes==p[0])[0]]+=1
        return top1/len(gt), top2/len(gt), CM
    
# Fonction de cross validation
def cross_validation_parzen(N,data_x,data_y,data_labels,model_func,classes,noyau,hmin=0,hmax=10,hstep=1):
    # Première étape : on divise les données en N datasets aléatoirement
    part = []
    for i in range(N):
        ids_app = [j for j in range(len(data_x)) if j%N != i]
        ids_val = [j for j in range(len(data_x)) if j%N == i]
        part.append({'app':[data_x[ids_app],data_y[ids_app],data_labels[ids_app]],
                     'val':[data_x[ids_val],data_y[ids_val],data_labels[ids_val]]})
    h_results = []
    for h in progressbar.progressbar(np.arange(hmin,hmax,hstep)):
        if h == 0:
            h = 1e-4
        h_top1 = []
        for i in range(N):
            model = model_func(classes,noyau,h)
            app_x,app_y,app_labels = part[i]['app']
            val_x,val_y,val_labels = part[i]['val']
            model.learning(app_x,app_y,app_labels)
            top1,__,__ = model.test(val_x,val_y,val_labels,print_results=False)
            h_top1.append(top1)
        h_results.append((h,np.mean(h_top1)))
    return h_results

def vote_majoritaire(k_classes,classes):
    counts = np.zeros((len(classes)))
    for i,c in enumerate(classes):
        counts[i] = k_classes.count(c)
    ind = np.argsort(counts)[::-1]
    return np.array(classes)[ind].tolist()

def vote_unanime(k_classes,classes):
    if len(np.unique(k_classes))==1:
        return k_classes[0]
    else:
        return None
    
class KPPV(Model):
    # Définition du constructeur
    def __init__(self,classes,vote='majoritaire',k=1):
        super(KPPV, self).__init__(classes)
        self.k = k
        assert vote in ['unanime','majoritaire']
        self.vote = vote
        if self.vote == 'unanime':
            self.vote_func = vote_unanime
        else:
            self.vote_func = vote_majoritaire
    
    # Définition de la fonction d'apprentissage
    def learning(self,app_x,app_y,app_labels):
        self.app_points = np.stack([app_x,app_y]).T
        self.app_labels = app_labels.to_numpy()
    
    def distance(self,A,B):
        return (A-B).T@(A-B)
    
    # Définition de la classe de prédiction
    def prediction(self,x,y):
        N = len(self.app_labels)
        point = np.array([x,y])
        labels = self.app_labels.copy()
        distances = np.array([self.distance(point,self.app_points[i,:]) for i in range(N)])
        k_classes_selec = []
        for i in range(self.k):
            ind_dist_min = np.argmin(distances)
            k_classes_selec.append(labels[ind_dist_min])
            distances = np.delete(distances,ind_dist_min)
            labels = np.delete(labels,ind_dist_min)
        return self.vote_func(k_classes_selec,self.classes)
    
    def metrics(self,gt,pred):
        if self.vote == 'majoritaire':
            return super(KPPV,self).metrics(gt,pred)
        else:
            CM = np.zeros((len(self.classes),len(self.classes)+1),dtype=np.uint16)
            top1 = 0
            top2 = 0
            for g,p in zip(gt,pred):
                if p is None:
                    CM[np.where(self.classes==g)[0],-1]+=1
                else:
                    if g == p:
                        top1+=1
                    CM[np.where(self.classes==g)[0],np.where(self.classes==p)[0]]+=1
            return top1/len(gt), top1/len(gt), CM
        
def cross_validation_KPPV(N,data_x,data_y,data_labels,model_func,classes,vote_func,hmin=1,hmax=10,hstep=1):
    # Première étape : on divise les données en N datasets aléatoirement
    part = []
    for i in range(N):
        ids_app = [j for j in range(len(data_x)) if j%N != i]
        ids_val = [j for j in range(len(data_x)) if j%N == i]
        part.append({'app':[data_x[ids_app],data_y[ids_app],data_labels[ids_app]],
                     'val':[data_x[ids_val],data_y[ids_val],data_labels[ids_val]]})


    h_results = []
    for h in progressbar.progressbar(np.arange(hmin,hmax,hstep)):
        if h==0:
            continue
        h_top1 = []
        for i in range(N):
            model = model_func(classes,vote_func,h)
            app_x,app_y,app_labels = part[i]['app']
            val_x,val_y,val_labels = part[i]['val']
            model.learning(app_x,app_y,app_labels)
            top1,__,__ = model.test(val_x,val_y,val_labels,print_results=False)
            h_top1.append(top1)
        h_results.append((h,np.mean(h_top1)))
    return h_results