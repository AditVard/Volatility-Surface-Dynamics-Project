**Volatility** **Surface** **Dynamics** **and** **Forecasting**

**Project** **Overview**

**This** **project** **explores** **the** **evolution** **of**
**implied** **volatility** **(IV)** **and** **local** **volatility**
**surfaces** **over** **time,** **with** **forecasting**
**capabilities.** **Using** **real-time** **options** **market**
**data** **(NIFTY** **index** **at** **9:15** **AM** **snapshots),**
**we** **analyze** **volatility** **surfaces** **and** **predict**
**future** **behavior** **using** **advanced** **statistical** **methods.**

**Key** **Concepts**

> **1.** **Fengler's** **Cubic** **Spline** **Smoothing-For**
> **constructing** **smooth** **IV** **surfaces**
>
> **2.** **Dupire's** **Local** **Volatility** **Model-For**
> **calculating** **local** **volatility** **surfaces**
>
> **3.** **Principal** **ComponentAnalysis** **(PCA)-For**
> **dimensionality** **reduction** **of** **volatility** **surfaces**
>
> **4.** **ARIMA** **Time** **Series** **Model-For** **forecasting**
> **future** **volatility** **patterns**

**Data**

> • **Real-time** **options** **market** **data** **for** **NIFTY**
> **index** **(9:15** **AM** **snapshots)**
>
> • **Weekly** **expiries** **with** **strikes** **ATM±5**
>
> • **Recent** **3** **expiries** **included** **in** **analysis**

**Project** **Highlights**

> **1.** **2D** **and** **3D** **visualization** **of** **implied**
> **volatility** **surfaces**
>
> **2.** **Computation** **and** **visualization** **of** **Dupire**
> **local** **volatility** **surfaces**
>
> **3.** **Daily** **Greeks** **generation** **for** **multiple**
> **strikes** **and** **expiries**
>
> **4.** **PCA** **decomposition** **of** **IV** **surfaces**
>
> **5.** **Storage** **ofIV** **surfacesin** **npz** **files** **for**
> **future** **analysis**
>
> **6.** **ARIMA** **forecasting** **of** **PCA** **components**
>
> **7.** **Comparison** **between** **predicted** **and** **actual**
> **IV** **surfaces**

**Project** **Structure**

**1.** **Surface** **Plotting**

> • **plot_iv_surface_3d.py-3D** **plots** **of** **IV** **surfaces**
>
> • **fengler_animate_iv_surface.py-Animation** **of** **Fengler's**
> **3D** **plots** **across** **dates**
>
> • **fenglera_singleday_iv_plot.py-Single-day** **Fengler** **surface**
> **plot**
>
> • **plot_fengler_2d.py-2D** **Fengler** **surface** **plots**
> **using** **cubic** **spline**
>
> • **plot_fengler_3d_iv_surface_singledate.py-Single-date** **3D**
> **Fengler** **surface** **plot**

**2.** **Data** **Collection** **and** **Greeks** **Generation**

> • **fetch_filtered_options.py-Fetches** **filtered** **options**
> **data** **(9:15** **AM)**
>
> • **generate_greeks.py-Generates** **Greeks** **for** **next** **3**
> **strikes**
>
> • **generate_greeks_onestriek.py-Generates** **Greeks** **for**
> **single** **expiry**
>
> • **spot_gather.py-Collects** **spot** **price** **data**
>
> • **save_fengleriv_3d_data.py-Saves** **Fengler** **IV** **data**
>
> • **load_stack_fengleriv.py-Loads** **and** **stacks** **Fengler**
> **IV** **data**
>
> • **saves_duprie_lv_data.py-Saves** **Dupire** **local**
> **volatility** **data**

**3.** **Dupire** **Local** **Volatility**

> • **dupire_local_vol_surface.py-Computes** **Dupire** **3D** **local**
> **volatility** **surfaces** **from** **Fengler** **IV**

**4.** **Principal** **Component** **Analysis**

> • **pca_fengler_surface_explained.py-PCA** **analysis** **of**
> **Fengler** **surfaces**
>
> • **pca_on_dupire_explained.py-PCA** **analysis** **of** **Dupire**
> **surfaces**
>
> • **pca_plot_3d_iv_surface_points.py-3D** **plot** **of** **IV**
> **surface** **points** **in** **PCA** **space**
>
> • **reconsturcts_iv_surface_using_pca.py-Reconstructs** **IV**
> **surfaces** **using** **PCA** **components**

**5.** **Forecasting** **and** **Evaluation**

> • **forecast_plot_pca1_arima.py-ARIMA** **forecasting** **for**
> **PC1**
>
> • **forecast_plot_pca2_arima.py-ARIMA** **forecasting** **for**
> **PC2**
>
> • **compare_iv_surface_witharima.py-Compares** **forecastedvs**
> **actual** **IV** **surfaces**

**Sample** **Output** **Files**

> **1.** **local_vol_surface_2025-04-22.npz-Local** **volatility**
> **surface** **data**
>
> **2.** **iv_surface_forecasted_2025-06-20.png-Forecasted** **IV**
> **surface** **visualization**
>
> **3.** **iv_surface_2025-04-22.npz-Real-time** **IV** **surface**
> **data**

**How** **to** **Run:**

**The** **project** **is** **modular** **,but** **some** **scripts** **do** **need**
**to** **use** **data** **generated** **by** **other** **files**

**Follow** **these** **steps** **to** **run** **the** **codes**

**1.** **Data** **Collection**

**Run** **these** **first** **to** **gather** **the** **base**
**option** **chain** **and** **spot** **data:**

> • **fetch_filtered_options.py–downloads** **filtered** **options**
> **chain** **(ATM** **±5)**
>
> • **spot_gather.py–gathers** **corresponding** **spot** **prices**

**2.** **Greeks** **Computation**

**Once** **data** **is** **fetched,** **compute** **Greeks:**

> • **generate_greeks.py–computes** **Delta,** **Vega,** **Gamma**
> **for** **all** **strikes** **(multi-expiry)**
>
> • **generate_greeks_onestriek.py–same,** **but** **for** **a**
> **single** **expiry**

**3.** **IV** **Surface** **Construction**

**Use** **cubic** **spline** **smoothing** **(Fengler** **method):**

> • **save_fengleriv_3d_data.py–saves** **Fengler** **IV** **surfaces**
> **(for** **each** **day)**
>
> • **plot_fengler_2d.py–2D** **visualization**
>
> • **plot_fengler_3d_iv_surface_singledate.py–single-day** **3D**
> **plot**
>
> • **fenglera_singleday_iv_plot.py–alternate** **3D** **plot** **(for**
> **one** **day)**
>
> • **fengler_animate_iv_surface.py–loops** **over** **all** **days**
> **and** **animates**

**4.** **Dupire** **Local** **Volatility**

**Calculate** **and** **store** **local** **volatility** **surfaces:**

> • **dupire_local_vol_surface.py–creates** **Dupire** **local** **vol**
> **surface** **from** **Fengler** **IV**
>
> • **saves_duprie_lv_data.py–saves** **Dupire** **vol** **surface**
> **data**

**5.** **PCA** **Decomposition**

**Reduce** **dimensionality** **using** **Principal** **Component**
**Analysis:**

> • **pca_fengler_surface_explained.py–PCA** **on** **Fengler** **IV**
> **data**
>
> • **pca_on_dupire_explained.py–PCA** **on** **Dupire** **local**
> **vol** **data**
>
> • **pca_plot_3d_iv_surface_points.py–3D** **plot** **of** **PCA**
> **component** **scores**

**6.** **Time-Series** **Forecasting**

**Forecast** **the** **PCA** **components:**

> • **forecast_plot_pca1_arima.py–forecasts** **PC1** **using**
> **ARIMA**
>
> • **forecast_plot_pca2_arima.py–forecasts** **PC2** **using**
> **ARIMA**

**7.** **Reconstruct** **&** **Evaluate**

**Rebuild** **IV** **surfaces** **using** **forecasted** **components**
**and** **compare:**

> • **reconsturcts_iv_surface_using_pca.py–reconstructs** **IV**
> **surfaces** **from** **forecasted** **PCA** **vectors**
>
> • **compare_iv_surface_witharima.py–compares** **forecastedvs**
> **actual** **IV** **surface**
>
> • **plot_iv_surface_3d.py–final** **3D** **plots** **for**
> **validation** **or** **presentation**

**Note:** **Most** **scripts** **assume** **data** **exists** **inside**
**the** **data/processed/folder.** **Modify** **the#** **CONFIG** **section**
**at** **the** **top** **of** **each** **script** **if** **your**
**paths** **are** **different.**

**Script** **can** **be** **run** **like** **this:**

**python** **scripts/plot_iv_surface_3d.py**

**Packages** **used:**

> • **numpy,pandas,matplotlib,seaborn** • **scipy,statsmodels,sklearn**
