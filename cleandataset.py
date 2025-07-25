import pandas as pd
df = pd.read_csv('C:/Users/sakina kulsum/sentiment_analysis_webapp/amazon_alexa.tsv', sep='\t')
print(df.shape)  # Should be (3150, ...)
print(df['verified_reviews'].isna().sum())  # Check for missing reviews
print(df['feedback'].isna().sum())  # Check for missing feedback
print(df['feedback'].value_counts())  # Check feedback values