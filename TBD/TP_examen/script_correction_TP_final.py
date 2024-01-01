import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import utils
import concurrent.futures

def compute(NUM,vars,path):
    print('yo')
    true_quals = [vars.loc[NUM-1]['qual1'],vars.loc[NUM-1]['qual2']]

    # Chargement de vos données app et dec
    data_app = pd.read_csv(path+f'data_app/data_app_{NUM}.csv',sep='\t')
    data_dec = pd.read_csv(path+f'data_dec/data_dec_{NUM}.csv',sep='\t')

    # Création des classifieurs distance minimum euclidiennes et Mahalanobis
    classes = np.unique(data_app['House'])

    CDE = utils.ClassifieurDistMin(classes)
    CDM = utils.ClassifieurDistMahalanobis(classes)

    # Apprentissage des classifieurs sur les données d'apprentissages (Courage et Kindness)
    CDE.learning(data_app['Courage'],data_app['Kindness'],data_app['House'])
    CDM.learning(data_app['Courage'],data_app['Kindness'],data_app['House'])

    # Evaluation des classifieurs sur les données d'évaluation
    CDE_results = CDE.test(data_dec['Courage'],data_dec['Kindness'],data_dec['House'],print_results=False)
    CDM_results = CDM.test(data_dec['Courage'],data_dec['Kindness'],data_dec['House'],print_results=False)

    # Cross validation à 3 dossiers pour déterminer les paramètres h optimaux pour un classifieur de Parzen avec noyau uniforme et noyau gaussien
    h_parzen = {'uniforme':None,'gaussien':None}
    for noyau in ['uniforme','gaussien']:
        h_results = utils.cross_validation_parzen(3,data_app['Courage'],data_app['Kindness'],data_app['House'],utils.Parzen,classes,noyau,0,10,1)
        h_max = max([p[1] for p in h_results])
        h_opti = [h for h in h_results if h[1]==h_max][0][0]
        h_results = utils.cross_validation_parzen(3,data_app['Courage'],data_app['Kindness'],data_app['House'],utils.Parzen,classes,noyau,int(h_opti-1),int(h_opti+1),0.1)
        h_max = max([p[1] for p in h_results])
        h_opti = [h for h in h_results if h[1]==h_max][0][0]
        h_parzen[noyau]=h_opti

    # Création des classifieurs de Parzen avec noyau uniforme et noyau gaussien
    PU = utils.Parzen(classes,'uniforme',h_parzen['uniforme'])
    PG = utils.Parzen(classes,'gaussien',h_parzen['gaussien'])

    # Apprentissage des classifieurs sur les données d'apprentissages (Courage et Kindness)
    PU.learning(data_app['Courage'],data_app['Kindness'],data_app['House'])
    PG.learning(data_app['Courage'],data_app['Kindness'],data_app['House'])

    # Evaluation des classifieurs sur les données d'évaluation
    PU_results = PU.test(data_dec['Courage'],data_dec['Kindness'],data_dec['House'],print_results=False)
    PG_results = PG.test(data_dec['Courage'],data_dec['Kindness'],data_dec['House'],print_results=False)

    # Cross validation à 3 dossiers pour déterminer les paramètres k optimaux pour un classifieur KPPV avec vote majoritaire et vote unanime
    k_KPPV = {'majoritaire':None,'unanime':None}
    for vote in ['majoritaire','unanime']:
        k_results = utils.cross_validation_KPPV(3,data_app['Courage'],data_app['Kindness'],data_app['House'],utils.KPPV,classes,vote,2,11,1)
        k_max = max([p[1] for p in k_results])
        k_opti = [k for k in k_results if k[1]==k_max][0][0]
        k_KPPV[vote]=k_opti

    # Création des classifieurs KPPV avec vote majoritaire et vote unanime
    KM = utils.KPPV(classes,'majoritaire',k_KPPV['majoritaire'])
    KU = utils.KPPV(classes,'unanime',k_KPPV['unanime'])

    # Apprentissage des classifieurs sur les données d'apprentissages (Courage et Kindness)
    KM.learning(data_app['Courage'],data_app['Kindness'],data_app['House'])
    KU.learning(data_app['Courage'],data_app['Kindness'],data_app['House'])

    # Evaluation des classifieurs sur les données d'évaluation
    KM_results = KM.test(data_dec['Courage'],data_dec['Kindness'],data_dec['House'],print_results=False)
    KU_results = KU.test(data_dec['Courage'],data_dec['Kindness'],data_dec['House'],print_results=False)

    top1_results_first = {'CDE':round(CDE_results[0],4),
                    'CDM':round(CDM_results[0],4),
                    'PU':round(PU_results[0],4),
                    'PG':round(PG_results[0],4),
                    'KM':round(KM_results[0],4),
                    'KU':round(KU_results[0],4)}

    top2_results_first = {'CDE':round(CDE_results[1],4),
                    'CDM':round(CDM_results[1],4),
                    'PU':round(PU_results[1],4),
                    'PG':round(PG_results[1],4),
                    'KM':round(KM_results[1],4),
                    'KU':round(KU_results[1],4)}

    param_optis_first = {'PU':round(h_parzen['uniforme'],1),
                         'PG':round(h_parzen['gaussien'],1),
                         'KM':round(k_KPPV['majoritaire'],1),
                         'KU':round(k_KPPV['unanime'],1)}

    CDE = utils.ClassifieurDistMin(classes)
    CDM = utils.ClassifieurDistMahalanobis(classes)

    # Apprentissage des classifieurs sur les données d'apprentissages (Courage et Kindness)
    CDE.learning(data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'])
    CDM.learning(data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'])

    # Evaluation des classifieurs sur les données d'évaluation
    CDE_results = CDE.test(data_dec[true_quals[0]],data_dec[true_quals[1]],data_dec['House'],print_results=False)
    CDM_results = CDM.test(data_dec[true_quals[0]],data_dec[true_quals[1]],data_dec['House'],print_results=False)

    # Cross validation à 3 dossiers pour déterminer les paramètres h optimaux pour un classifieur de Parzen avec noyau uniforme et noyau gaussien
    h_parzen = {'uniforme':None,'gaussien':None}
    for noyau in ['uniforme','gaussien']:
        h_results = utils.cross_validation_parzen(3,data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'],utils.Parzen,classes,noyau,0,10,1)
        h_max = max([p[1] for p in h_results])
        h_opti = [h for h in h_results if h[1]==h_max][0][0]
        h_results = utils.cross_validation_parzen(3,data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'],utils.Parzen,classes,noyau,int(h_opti-1),int(h_opti+1),0.1)
        h_max = max([p[1] for p in h_results])
        h_opti = [h for h in h_results if h[1]==h_max][0][0]
        h_parzen[noyau]=h_opti

    # Création des classifieurs de Parzen avec noyau uniforme et noyau gaussien
    PU = utils.Parzen(classes,'uniforme',h_parzen['uniforme'])
    PG = utils.Parzen(classes,'gaussien',h_parzen['gaussien'])

    # Apprentissage des classifieurs sur les données d'apprentissages (Courage et Kindness)
    PU.learning(data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'])
    PG.learning(data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'])

    # Evaluation des classifieurs sur les données d'évaluation
    PU_results = PU.test(data_dec[true_quals[0]],data_dec[true_quals[1]],data_dec['House'],print_results=False)
    PG_results = PG.test(data_dec[true_quals[0]],data_dec[true_quals[1]],data_dec['House'],print_results=False)

    # Cross validation à 3 dossiers pour déterminer les paramètres k optimaux pour un classifieur KPPV avec vote majoritaire et vote unanime
    k_KPPV = {'majoritaire':None,'unanime':None}
    for vote in ['majoritaire','unanime']:
        k_results = utils.cross_validation_KPPV(3,data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'],utils.KPPV,classes,vote,2,11,1)
        k_max = max([p[1] for p in k_results])
        k_opti = [k for k in k_results if k[1]==k_max][0][0]
        k_KPPV[vote]=k_opti

    # Création des classifieurs KPPV avec vote majoritaire et vote unanime
    KM = utils.KPPV(classes,'majoritaire',k_KPPV['majoritaire'])
    KU = utils.KPPV(classes,'unanime',k_KPPV['unanime'])

    # Apprentissage des classifieurs sur les données d'apprentissages (Courage et Kindness)
    KM.learning(data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'])
    KU.learning(data_app[true_quals[0]],data_app[true_quals[1]],data_app['House'])

    # Evaluation des classifieurs sur les données d'évaluation
    KM_results = KM.test(data_dec[true_quals[0]],data_dec[true_quals[1]],data_dec['House'],print_results=False)
    KU_results = KU.test(data_dec[true_quals[0]],data_dec[true_quals[1]],data_dec['House'],print_results=False)

    top1_results_true = {'CDE':round(CDE_results[0],4),
                    'CDM':round(CDM_results[0],4),
                    'PU':round(PU_results[0],4),
                    'PG':round(PG_results[0],4),
                    'KM':round(KM_results[0],4),
                    'KU':round(KU_results[0],4)}

    top2_results_true = {'CDE':round(CDE_results[1],4),
                    'CDM':round(CDM_results[1],4),
                    'PU':round(PU_results[1],4),
                    'PG':round(PG_results[1],4),
                    'KM':round(KM_results[1],4),
                    'KU':round(KU_results[1],4)}

    param_optis_true = {'PU':round(h_parzen['uniforme'],1),
                         'PG':round(h_parzen['gaussien'],1),
                         'KM':round(k_KPPV['majoritaire'],1),
                         'KU':round(k_KPPV['unanime'],1)}

    return true_quals,top1_results_first,top2_results_first,param_optis_first,top1_results_true,top2_results_true,param_optis_true

def main():
    path = "DATA_TP_FINAL/"

    vars = pd.read_csv(path+'vars.csv',sep='\t')

    tab_results = pd.DataFrame(columns=['qual1','qual2',
                                        'CDE_first_top1','CDE_first_top2','CDM_first_top1','CDM_first_top2',
                                        'PU_first_h','PU_first_top1','PU_first_top2','PG_first_h','PG_first_top1','PG_first_top2',
                                        'KM_first_h','KM_first_top1','KM_first_top2','KU_first_h','KU_first_top1','KU_first_top2',
                                        'CDE_true_top1','CDE_true_top2','CDM_true_top1','CDM_true_top2',
                                        'PU_true_h','PU_true_top1','PU_true_top2','PG_true_h','PG_true_top1','PG_true_top2',
                                        'KM_true_h','KM_true_top1','KM_true_top2','KU_true_h','KU_true_top1','KU_true_top2'],
                              index=list(range(1,31)))


    results = []

    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        for NUM in range(1,11):
            results.append(executor.submit(compute,NUM,vars,path))
            
    print(results) 


    #tab_results.loc[NUM-1] = [true_quals[0],true_quals[1],
    #                          top1_results_first['CDE'],top2_results_first['CDE'],top1_results_first['CDM'],top2_results_first['CDM'],
    #                          param_optis_first['PU'],top1_results_first['PU'],top2_results_first['PU'],param_optis_first['PG'],top1_results_first['PG'],top2_results_first['PG'],
    #                          param_optis_first['KM'],top1_results_first['KM'],top2_results_first['KM'],param_optis_first['KU'],top1_results_first['KU'],top2_results_first['KU'],
    #                          top1_results_true['CDE'],top2_results_true['CDE'],top1_results_true['CDM'],top2_results_true['CDM'],
    #                          param_optis_true['PU'],top1_results_true['PU'],top2_results_true['PU'],param_optis_true['PG'],top1_results_true['PG'],top2_results_true['PG'],
    #                          param_optis_true['KM'],top1_results_true['KM'],top2_results_true['KM'],param_optis_true['KU'],top1_results_true['KU'],top2_results_true['KU']]
    
    
if __name__ == '__main__':
    main()