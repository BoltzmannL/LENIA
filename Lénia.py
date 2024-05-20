# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:11:30 2024

@author: toset
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sp

#définition des fonctions
dt = 0.1
R = 13
x = np.linspace(-1, 1, 2*R+1)
y = np.linspace(-1, 1, 2*R+1)
[X, Y] = np.meshgrid(x,y)
dist = np.sqrt((X)**2+(Y)**2)

#filtre
def gauss(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)

mu = 0.5
sigma = 0.15
mask = gauss(dist, mu, sigma) 
mask = mask / np.sum(mask)
#plt.imshow(mask, cmap = 'inferno')
#plt.title('filtre de rayon: R = {}'.format(R))
#plt.colorbar()

#fonction de croissance
def growth (u):
    mu_g = 0.15
    sigma_g = 0.015
    return -1 + 2*gauss(u, mu_g, sigma_g)

#xt = np.linspace(0, 0.3, 100)
#plt.plot(xt, growth(xt), color = 'b')
#plt.title('fonction de croissance')
#plt.xlabel('somme des valeurs voisines')
#plt.ylabel('taux de croissance')

#Convolution du filtre et de la fonction de croissance
def convo(i):
    global A 
    S = sp.signal.convolve2d(A, mask, mode = 'same', boundary = bond)
    A = A + dt*growth(S) #+ dt*(i/10)
    A = np.clip(A, 0, 1)
    #print(i)
    if aff == 'animé':
        img.set_array(A)
        return img
    else:
        return A
    
#Interface utilisateur
conf = input('quelle configuration désirez vous ? orbium ? point ? aléatoire ?\n')
input_test = ['orbium', 'point', 'aléatoire']
if conf not in input_test[:]:
    raise Exception('Configuration invalide, veuillez choisir entre orbium, point ou aléatoire')

N = input('quelle taille de grille désirez vous ?\n')
try:
    N = int(N)
except:
    raise Exception('Veuillez entrer un nombre entier')
if N < 30:
    raise Exception('La grille doit être de taille 30 minimum')

if N < 100 and conf == 'point' and conf == 'aléatoire':
    raise Exception('la grille ne peut être plus petite que 100 pour cette configuration')

bond = input('Veuillez choisir une condition aux bords entre "fill", "wrap" et "symm"\n')
bond_test = ['fill', 'wrap', 'symm'] 
if bond not in bond_test[:]:
    raise Exception('Conditions aux bords invalide')
if conf == 'orbium' and bond == 'fill':
    raise Exception('Cette configuration nécessite une condition aux bords "wrap" ou "symm" pour fonctionner')
    
aff = input('Voulez vous un affichage animé ou par étape ?\n')
aff_test = ['par étape', 'animé']
if aff not in aff_test[:]:
    raise Exception('Affichage invalide')
    
#Initialisation 
if conf == 'orbium':
    if R != 13:
        raise Exception('R doit être égal à 13 pour cette configuration')
    A = np.zeros((N, N))
    orbium = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])
    x_o = int(N/3)
    y_o = int(N/3)
    #print(orbium.shape)
    #print(orbium.shape[0])
    A[x_o:(x_o + orbium.shape[0]), y_o:(y_o + orbium.shape[1])] = orbium #+ np.random.rand(20,20)*0.5
    #plt.imshow(A, cmap='inferno') 
elif conf == 'point':
    R_p = 36
    xp = np.arange(-N//2, N//2)
    yp = np.arange(-N//2, N//2)
    Xp, Yp = np.meshgrid(xp, yp)
    A = np.exp(-0.5*((Xp**2 + Yp**2)/R_p**2))

elif conf == 'aléatoire' :
    A = np.random.rand(N, N)

#Affichage par étape
if aff == 'par étape':
    plt.close('all')
    plt.figure('Lénia')
    plt.imshow(A, cmap = 'inferno')
    plt.colorbar()
    plt.title('Configuration : {}'.format(conf), fontsize = 16)
    plt.xlabel('Rang : 0', fontsize = 16)
    convo(A)
    for j in range(101):
        A = convo(A)
        if j == 20:
            plt.figure('Lénia 20')
            plt.imshow(A, cmap = 'inferno')
            plt.colorbar()
            plt.title('Configuration : {}'.format(conf), fontsize = 16)
            plt.xlabel('Rang : {}'.format(j), fontsize = 16)
        if j == 50:
            plt.figure('Lénia 50')
            plt.imshow(A, cmap = 'inferno')
            plt.colorbar()
            plt.title('Configuration : {}'.format(conf), fontsize = 16)
            plt.xlabel('Rang : {}'.format(j), fontsize = 16)
        if j == 100:
            plt.figure('Lénia 100')
            plt.imshow(A, cmap = 'inferno')
            plt.colorbar()
            plt.title('Configuration : {}'.format(conf), fontsize = 16)
            plt.xlabel('Rang : {}'.format(j), fontsize = 16)
            
#Affichage animé
else:
    fig = plt.figure('Lénia', figsize = (12,12))
    #img = plt.imshow(A, cmap = 'inferno')
    img = plt.imshow(A, cmap ='inferno', interpolation = 'bicubic')
    plt.title('Configuration : {}'.format(conf), fontsize = 16)
    ani = animation.FuncAnimation(fig, convo, frames=50, interval=50)

