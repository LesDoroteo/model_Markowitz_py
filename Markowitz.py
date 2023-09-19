import pandas as pd
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import yfinance as yfin
import statistics
# Importamos data desde Yahoo.finance con la libreria de yfinance y pandas_datareader
#Se importo 5 activos que pertenecen a la categoría de TI
cartera = ['WMT','KO','KHC','TSLA','AAPL']

data = pd.DataFrame()
yfin.pdr_override()

#Se extraer información de yahoo finance desde el 2014 al 2019 para evitar el ruido en la información
data = wb.get_data_yahoo(['WMT','KO','KHC','TSLA','AAPL'] , start='2014-1-1', end='2019-1-1')['Adj Close']
#print(data.head())

#Generamos los retornos por activo y convertimos en porcentajes    
log_return = np.log(1+data.pct_change())
#print(log_return.head())

# 1. Hallamos los pesos por activos
# 2. La suma de los pesos deben dar 1 o 100%
# 3. se halla el retorno de los portafolios (suma producto del rendimiento con los pesos)
# 4. Se halla la varianza de la cartera (la raíz de la matriz de la cov por los pesos y la transpuesta de sus pesos)
# 5. Se genera los retornos correlacionados o ponderados (port_returns)
# 6. Se genera la volatilidad de la cartera (port_vols)

port_returns = []
port_vols = []

for i in range (100000):
    num_assets = len(cartera)
    weights = np.random.random(num_assets)
    weights_1= weights/np.sum(weights)
    port_ret = np.sum(log_return.mean()*weights_1)*252
    port_var = np.sqrt(np.dot(weights_1.T, np.dot(log_return.cov()*252, weights_1)))
    port_returns.append(port_ret)
    port_vols.append(port_var)
    Sharpe_ind = port_ret/port_var

#print(weights_1)
#print(port_ret)
#print(port_var)
#print(Sharpe_ind)

# Definimos la función del portafolio

def portfolio_stats(weights_1, log_return):
    port_ret = np.sum(log_return.mean()*weights_1)
    port_var = np.sqrt(np.dot(weights_1.T, np.dot(log_return.cov(), weights_1)))
    Sharpe_ind = port_ret/port_var
    return {'Return': port_ret, 'Volatility': port_var, 'Sharpe': Sharpe_ind}

# Optimizamos el ratio Sharpe con Scipy ya que no ofrece la función maximizar, se realiza la inversa de la funcióo portfolio_Stats que es el negativo

def minimize_sharpe(weights_1, log_return):
    return -portfolio_stats(weights_1, log_return)['Sharpe']

# Generamos las matrices de los resultados de la cartera, las volatilidades y definimos Sharpe
port_returns = np.array(port_returns)
port_vols = np.array(port_vols)
Sharpe = port_returns/port_vols

    #print(Sharpe)
# Optimizamos la cartera

max_sr_vol = port_vols[Sharpe.argmax()]
max_sr_ret = port_returns[Sharpe.argmax()]
#print(max_sr_ret)

# especificando parametros para la función optimize.minimize

constraints = ({'type' : 'eq' , 'fun' : lambda x: np.sum(x) -1})
bounds  = tuple((0,1) for x in range(num_assets))
initializer = num_assets*[1./num_assets,]

optimal_sharpe = optimize.minimize( minimize_sharpe, initializer, method='SLSQP' , args= (log_return,), bounds = bounds, constraints  = constraints)
optimal_sharpe_weights = optimal_sharpe['x'].round(4)
optimal_stats = portfolio_stats(optimal_sharpe_weights, log_return)

print("Pesos óptimos de la cartera: ", list(zip(cartera, list(optimal_sharpe_weights*100))))
print("Retorno óptimo de la cartera: ", round(optimal_stats['Return']*100,4))
print("Volatilidad óptima de la cartera: ", round(optimal_stats['Volatility']*100,4))
print("Ratio Sharpe óptima de la cartera: ", round(optimal_stats['Sharpe'],4))

# Gráfico de la frontera eficiente

plt.figure(figsize=(10,6))
plt.scatter(port_vols, port_returns,c = port_returns/port_vols)
plt.scatter(max_sr_vol, max_sr_ret, c='red' , s=30)
plt.colorbar(label = 'Ratio Sharpe (rf=0)')
plt.xlabel('Volatilidad de la cartera')
plt.ylabel('Retorno de la cartera')
plt.show()