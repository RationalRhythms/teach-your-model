import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class DataProcessor:
    def __init__(self):
        self.category_mapping = {
            'comp.graphics': 'technology', 'comp.os.ms-windows.misc': 'technology',
            'comp.sys.ibm.pc.hardware': 'technology', 'comp.sys.mac.hardware': 'technology',
            'comp.windows.x': 'technology', 'sci.space': 'science', 'sci.med': 'science',
            'sci.electronics': 'science', 'sci.crypt': 'science', 'rec.sport.baseball': 'recreation',
            'rec.sport.hockey': 'recreation', 'rec.autos': 'recreation', 'rec.motorcycles': 'recreation',
            'talk.politics.misc': 'politics', 'talk.politics.guns': 'politics', 
            'talk.politics.mideast': 'politics', 'talk.religion.misc': 'politics',
            'misc.forsale': 'forsale', 'alt.atheism': 'politics'
        }
        
        self.super_categories = ['technology', 'science', 'recreation', 'politics', 'forsale']
    
    def load_and_regroup_data(self,initial_labeled_ratio=0.5):
        """Load data and return analysis"""
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        
        df = pd.DataFrame({
            'text': newsgroups.data,
            'original_label': newsgroups.target,       # returns numeric label
            'original_category': [newsgroups.target_names[i] for i in newsgroups.target]   # map numeric label to name
        })
        
        df['super_category'] = df['original_category'].map(self.category_mapping)
        df['super_label'] = df['super_category'].map(
            {cat: i for i, cat in enumerate(self.super_categories)}
        )
        
        df = df[df['super_category'].notna() & (df['text'].str.len() > 50)]
        
        labeled_df, unlabeled_df = train_test_split(
            df, 
            train_size=initial_labeled_ratio, 
            random_state=42, 
            stratify=df['super_category'] # preserve original distribution
        )
        
        unlabeled_pool = unlabeled_df.drop(['super_category', 'super_label'], axis=1).copy()
        
        return labeled_df, unlabeled_pool
    
    def generate_distribution_report(self, df):
        """Generate comprehensive distribution analysis"""
        print("CATEGORY DISTRIBUTION ANALYSIS")
        print("\n")
        
        super_counts = df['super_category'].value_counts()
        print("\n1. SUPER CATEGORY DISTRIBUTION:")
        for category, count in super_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {category:12} : {count:4} documents ({percentage:5.1f}%)")
    
        
        df['text_length'] = df['text'].str.len()
        print(f"\n3. TEXT LENGTH STATISTICS:")
        print(f"   Average length: {df['text_length'].mean():.0f} characters")
        print(f"   Min length: {df['text_length'].min()} characters")
        print(f"   Max length: {df['text_length'].max()} characters")
        
        return super_counts


if __name__ == "__main__" :
    processor = DataProcessor()
    df,_ = processor.load_and_regroup_data()
    distribution = processor.generate_distribution_report(df)