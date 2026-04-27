import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

    # ── 1. LOAD DATA ─────────────────────────────
df = pd.read_csv('titanic_output.csv')

    # ── 2. HANDLE MISSING VALUES ─────────────────
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # ── 3. CONVERT CATEGORY ──────────────────────
df['Survived'] = df['Survived'].map({0:'Not Survived', 1:'Survived'})

    # ── 4. BASIC INFO (IMPORTANT FOR MARKS) ──────
print(df.info())
print(df.describe())

    # ── 5. STYLE ────────────────────────────────
sns.set_theme(style='whitegrid')
sns.set_palette('viridis')

    # ── 6. HISTOGRAM + KDE ───────────────────────
sns.histplot(data=df, x='Age', kde=True, hue='Survived')
plt.title('Age Distribution by Survival')
plt.show()

    # ── 7. BOX PLOT ──────────────────────────────
sns.boxplot(data=df, x='Survived', y='Fare')  
plt.title('Fare by Survival')
plt.show()

    # ── 8. VIOLIN PLOT ───────────────────────────
sns.violinplot(data=df, x='Survived', y='Age')
plt.title('Age Distribution by Survival (Violin)')
plt.show()

    # ── 9. HEATMAP ───────────────────────────────
sns.heatmap(df.select_dtypes(include='number').corr(),
                annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

    # ── 10. PAIRPLOT ─────────────────────────────
sns.pairplot(
        df[['Age','Fare','Pclass','SibSp','Parch','Survived']],
        hue='Survived',
        diag_kind='kde'
    )
plt.show()

    # ── 11. EXTRA (REGRESSION PLOT - IMPRESSIVE) ─
sns.regplot(data=df, x='Age', y='Fare', scatter_kws={'alpha':0.3})
plt.title('Age vs Fare Relationship')
plt.show()