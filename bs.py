#Fill missing employee count with the mean of the column
df['Employees'].fillna(df['Employees'].mean(), inplace=True)

# Fill missing Companynames with 'Unknown'
df['Company Name'].fillna('Unknown', inplace=True)

# Fill missing Founded with 'Unknown'
df['Founded'].fillna('Unknown', inplace=True)

# Remove duplicates based on all columns
df = df.drop_duplicates()

df = pd.read_csv('data.csv')


# Convert Founded to a standard format
df['Founded'] = pd.to_datetime(df['Founded'], errors='coerce').dt.strftime('%m-%d-%Y')

# Remove dollar signs and commas from Revenue and convert to integers
df['Revenue'] = df['Revenue'].str.replace(r'[\$,]', '', regex=True).fillna('0').astype(int)


# Check for any remaining missing values
print(df.isnull().sum())

# Check the first few rows of the cleaned data
print(df.head())


# Verify date formats are consistent
print(df['Founded'].head())

# Check the range of Employees to ensure reasonable values
print(df['Employees'].describe())

# Compare the number of rows before and after cleaning
print(f"Original rows: {original_df.shape[0]}, Cleaned rows: {df.shape[0]}")
