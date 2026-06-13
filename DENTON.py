import numpy as np
import matplotlib.pyplot as plt
# import xlwings as xw
import pandas as pd
from pandas import DataFrame
# import matplotlib.pyplot as plt
import matplotlib.tri as mtri
# from matplotlib.ticker import MultipleLocator
# from matplotlib.collections import PolyCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
# from dataclasses import dataclass
# from typing import Sequence
from matplotlib.colors import BoundaryNorm
from scipy.interpolate import griddata
from scipy.integrate import simpson

### DEFAULTS ######################

_x = np.linspace(0, np.pi, 360)

###################################
#   D E N T O N     M E T H O D   #
###################################

def capacity_to_triad(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mom = arr[:,0]
    ang = arr[:,1]
    ang = np.deg2rad(arr[:,1])

    mx = np.sum(mom * np.cos(ang) ** 2)
    my = np.sum(mom * np.sin(ang) ** 2)
    mxy = -np.sum(mom * np.sin(ang) * np.cos(ang))
    return np.array([mx, my, mxy], dtype=float)

def create_moment_field(triad: np.ndarray, 
                 angle: float | np.ndarray = _x) -> np.ndarray:
    Mx, My, Mxy = triad
        
    s = np.sin(angle)
    c = np.cos(angle)
    return Mx * c ** 2 + My * s ** 2 - 2 * Mxy * s * c

def calculate_gamma_theta(MN: np.ndarray,
                          MR: np.ndarray,
                          X: np.ndarray = _x,
                          *,
                          eps= 1e-12,
                          tol= 1e-12):
    
    # jeżeli funkcja nośności MR < 0 
    if np.any((MR < 0 ) & (MN > eps)):
        return 1000, 0.
    mask = (MN > 0) & (MR >= 0)
    if not np.any(mask):
        return 1000, 0.
    ratio = MR[mask] / MN[mask]
    iloc = np.argmin(ratio)
    gamma = ratio[iloc]
    theta = X[mask][iloc]
    
    #walidacja
    if np.any(gamma * MN > MR + tol):
        den = np.clip(MN, eps, None)
        gamma = np.min((MR + tol) / den)
    return float(gamma), float(theta)

#### pandas help functions #####

def nodes_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={
        "X(m)": "x",
        "Y(m)": "y",
        "Z(m)": "z",
    }, inplace=True)
    df['Node'] = df['Node'].astype("Int64")
    df.reset_index()
    return df

def elements_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={
        'Element': 'Elem'
    }, inplace=True)
    df = df.loc[df["Type"] == "PLATE"]
    df = df[['Elem','Node1', 'Node2', 'Node3', 'Node4', 'Node5', 'Node6', 'Node7', 'Node8']]
    df = df.astype("Int64")
    # df.reset_index(inplace=True)
    return df

def results_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    headers = ['Elem', 'Load', 'Node', 'Mxx (kN·m/m)', 'Myy (kN·m/m)', 'Mxy (kN·m/m)',
        'Vxx (kN/m)', 'Vyy (kN/m)']
    rename = {
        'Mxx (kN·m/m)': 'mxx', 
        'Myy (kN·m/m)': 'myy', 
        'Mxy (kN·m/m)': 'mxy',
        'Vxx (kN/m)': 'vxx', 
        'Vyy (kN/m)': 'vyy'
    }
    df = df.loc[:, headers]
    df.rename(columns=rename, inplace=True)
    df['Elem'] = df['Elem'].astype("Int64")
    df['Node'] = df['Node'].astype("Int64")
    return df

def _calc_gamma(r, MR):
    mx = r['mxx']
    my = r['myy']
    mxy = r['mxy']
    MN = create_moment_field([mx,my,mxy])
    gamma = calculate_gamma_theta(MN,MR)[0]
    return gamma

##########################################################################################

def integrate_between_points(x,y,z,P1,P2, type_of_interpolation:str = 'linear') -> float:
    P1 = np.asarray(P1,dtype=float)
    P2 = np.asarray(P2,dtype=float)
    L = np.linalg.norm(P2-P1)
    points = np.column_stack((x,y))

    t = np.linspace(0, 1, 1000)
    line = P1 + np.outer(t, (P2 - P1))
    V = griddata(points, z, line, type_of_interpolation, fill_value=0)
    integral = simpson(V, t)
    return float(integral)

def cutting_diagram(x,y,z,P1,P2, type_of_interpolation:str = 'linear') -> float:
    P1 = np.asarray(P1,dtype=float)
    P2 = np.asarray(P2,dtype=float)
    L = np.linalg.norm(P2-P1)
    points = np.column_stack((x,y))

    t = np.linspace(0, 1, 1000)
    line = P1 + np.outer(t, (P2 - P1))
    V = griddata(points, z, line, type_of_interpolation, fill_value=0)
    return np.column_stack((L*t,V))

def cutting_line_plot(results: pd.DataFrame,
                      cutting_lines: list[dict[str, int, float]],
                      colors: list[dict[str, int, float]],
                      labels: list[dict[str, int, float]],
                      loads: list,
                      type_of_interpolation: str='linear'):

    loads_number = len(loads)
    
    fig, ax = plt.subplots(loads_number,1, figsize=(8,3.75 * loads_number))

    for i, load in enumerate(loads):
        mask = results['Load'] == load
        
        x = results.loc[mask, 'x'].to_numpy(dtype=float)
        y = results.loc[mask, 'y'].to_numpy(dtype=float)
        z = results.loc[mask, 'vxx'].to_numpy(dtype=float)
            
        for seg in cutting_lines:
            arr = cutting_diagram(x, y, z, 
                                seg['P1'], 
                                seg['P2'], 
                                type_of_interpolation)

            ax[i].plot(
                arr[:,0], arr[:,1],
                color=colors.get(seg['name'], None),
                lw=2,
                label=labels.get(seg['name'], seg['name'])
            )
            
        ax[i].grid(linestyle=':')
        ax[i].set_title(str(load))
        ax[i].set_xlabel('Distance from Pt. 1: x [m]')          
        ax[i].set_ylabel('Shear: Vxx [kN/m]')     
        ax[i].set_xlim(arr[:,0].min(), arr[:,0].max())
  
    legend_handles = [
        Line2D([0], [0], color=colors.get(k, 'black'), lw=2, label=v)
        for k, v in labels.items()
    ]

    fig.legend(handles=legend_handles, loc='upper center', ncol=min(4, len(legend_handles)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.999])  # leave space at the top for the legend
    return fig

def Vxx_average(results: DataFrame,
                cutting_lines: list[dict[str, int, float]],
                ) -> DataFrame:
    
    loads = results['Load'].unique().tolist()

    rows = []
    for load in loads:
        mask = results['Load'] == load
        x = (results.loc[mask, 'x'].to_numpy(dtype=float))
        y = (results.loc[mask, 'y'].to_numpy(dtype=float))    
        z = (results.loc[mask, 'vxx'].to_numpy(dtype=float))
        row = {"Load": load}
        for seg in cutting_lines:
            row[seg["name"]] = integrate_between_points(
                x, y, z,
                P1=seg["P1"], P2=seg["P2"],
                type_of_interpolation='linear'
            )
        rows.append(row)

    df_int = pd.DataFrame(rows).set_index("Load")
    df_int['Vxx_max d from support'] = df_int[['d_L', 'd_R']].abs().max(axis=1)
    df_int['Vxx_max 3d from support'] = df_int[['3d_L',	'3d_R']].abs().max(axis=1)

    # df_int = df_int.round(1)
    return df_int

#########################################################################################

#### PLOTING #####

def contour_plot(x,y,z, *, cmin=None, cmax=None, title: str= "Contour plot"):
    tri = mtri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=(8,5), dpi=150)

    # --- USTAWIENIA KOLORÓW / NORMY ---
    cmin = min(z) if cmin is None else cmin
    cmax = max(z) if cmax is None else cmax
    n_levels = 12                        # ile przedziałów (np. 10 → 10 „kafli”)
    bounds = np.linspace(cmin, cmax, n_levels + 1)  # np. 0,1,2,...,10 (11 krawędzi → 10 przedziałów)
    norm = BoundaryNorm(bounds, ncolors=256, clip=False)  # clip=False → wartości <cmin i >cmax dostaną „extend”

    # --- WYPEŁNIENIE (MUSI dostać norm + levels=bounds) ---
    cntr = ax.tricontourf(
        tri, z,
        levels=bounds,          
        norm=norm,             
        cmap="turbo_r",             # 'turbo', 'viridis', 'JET_R'
        antialiased=True
    )

    # Linie konturów (spójne z bounds)
    ax.tricontour(
        tri, z,
        levels=bounds,
        colors="k",
        linewidths=0.5,
        alpha=0.5,
        norm=norm,              
        antialiased=True
    )

    cb = fig.colorbar(
        cntr, ax=ax, label="γ (gamma)",  
        boundaries=bounds,               
        ticks=bounds,                    
        extend="both"                    
    )

    ax.triplot(
        tri,
        color="0.5",        
        linewidth=0.3,
        alpha=0.5,
        zorder=0
    )
    
    ax.set_aspect("equal")
    ax.set_title(title)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    plt.tight_layout()
    return fig