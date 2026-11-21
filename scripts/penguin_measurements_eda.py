"""Basic EDA on the Palmer Penguins dataset (NumPy + Pandas, no machine learning).
Converted from Penguin_Measurements_EDA.ipynb. Run: python penguin_measurements_eda.py
Expects penguins_size.csv in the same folder; saves two charts next to the script."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)

# ---------- Load & inspect ----------
df = pd.read_csv('penguins_size.csv')
print('HEAD');  print(df.head())
print('\nTAIL'); print(df.tail())
print('\nSHAPE', df.shape)
print('\nINFO'); df.info()
print('\nDESCRIBE'); print(df.describe())
print('\nDUPLICATE ROWS:', df.duplicated().sum())

# ---------- Q1: count penguins by species and island ----------
print('\nQ1 species counts'); print(df['Species'].value_counts())
print('\nIsland counts');     print(df['Island'].value_counts())
print('\nSpecies x Island');  print(df.groupby(['Species', 'Island']).size())

# ---------- Q2: average body mass and flipper length by species ----------
print('\nQ2'); print(df.groupby('Species')[['Body Mass (g)', 'Flipper Length (mm)']].mean())

# ---------- Q3: species with the highest average body mass ----------
avg_body_mass = df.groupby('Species')['Body Mass (g)'].mean()
print('\nQ3 heaviest species:', avg_body_mass.idxmax(), '| average:', round(avg_body_mass.max(), 1))

# ---------- Q4: average bill (culmen) length by species ----------
print('\nQ4'); print(df.groupby('Species')['Culmen Length (mm)'].mean())

# ---------- Q5: missing values in sex and measurement columns ----------
print('\nQ5 missing values')
print(df[['Sex', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']].isnull().sum())

# ---------- Q6: 10 penguins with the greatest body mass ----------
print('\nQ6'); print(df.sort_values('Body Mass (g)', ascending=False).head(10))

# ---------- NumPy calculations ----------
mass = df['Body Mass (g)'].dropna().to_numpy()
print('\nNumPy mean   :', np.mean(mass))
print('NumPy median :', np.median(mass))
print('NumPy std    :', np.std(mass))

# ---------- Pandas filtering example ----------
print('\nGentoo penguins heavier than 5500 g:')
print(df[(df['Species'] == 'Gentoo penguin (Pygoscelis papua)') & (df['Body Mass (g)'] > 5500)])

# ---------- Chart 1: species-count bar chart ----------
counts = df['Species'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values, color=['#4c72b0', '#dd8452', '#55a868'])
plt.title('Penguin Count by Species'); plt.xlabel('Species'); plt.ylabel('Count')
plt.xticks(fontsize=7); plt.tight_layout()
plt.savefig('species_count.png', dpi=150); plt.show()

# ---------- Chart 2: body-mass histogram ----------
plt.figure(figsize=(6, 4))
plt.hist(mass, bins=20, color='#4c72b0', edgecolor='white')
plt.axvline(np.mean(mass), color='red', ls='--', label='Mean %.0f g' % np.mean(mass))
plt.title('Distribution of Body Mass'); plt.xlabel('Body mass (g)'); plt.ylabel('Frequency')
plt.legend(); plt.tight_layout()
plt.savefig('body_mass_hist.png', dpi=150); plt.show()

# ---------- Three observations ----------
# 1. Gentoo has the highest average body mass (5076.0 g), about 1,350 g above Adelie and Chinstrap.
# 2. Average bill length: Chinstrap 48.83 mm > Gentoo 47.50 mm > Adelie 38.79 mm.
# 3. Missing values: Sex 11, and 2 each in Culmen Length, Culmen Depth, Flipper Length and Body Mass
#    (the same 2 rows have no measurements at all).
