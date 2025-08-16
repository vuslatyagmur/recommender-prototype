import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class RecommendationEngine:
    """
    Advanced recommendation engine that uses multiple factors to generate personalized recommendations.
    """
    
    def __init__(self, items_df: pd.DataFrame):
        """
        Initialize the recommendation engine with items data.
        
        Args:
            items_df: DataFrame containing item information
        """
        self.items_df = items_df.copy()
        self.similarity_cache = {}
        
    def calculate_content_similarity(self, item1: str, item2: str) -> float:
        """
        Calculate content-based similarity between two items.
        
        Args:
            item1, item2: Names of items to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        cache_key = tuple(sorted([item1, item2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Get item details
        item1_data = self.items_df[self.items_df['name'] == item1]
        item2_data = self.items_df[self.items_df['name'] == item2]
        
        if item1_data.empty or item2_data.empty:
            return 0.0
        
        item1_row = item1_data.iloc[0]
        item2_row = item2_data.iloc[0]
        
        similarity = 0.0
        factors = 0
        
        # Genre similarity
        if 'genre' in self.items_df.columns:
            if pd.notna(item1_row['genre']) and pd.notna(item2_row['genre']):
                if item1_row['genre'] == item2_row['genre']:
                    similarity += 0.4
                factors += 0.4
        
        # Year similarity (for time-based preferences)
        if 'year' in self.items_df.columns:
            if pd.notna(item1_row['year']) and pd.notna(item2_row['year']):
                year_diff = abs(item1_row['year'] - item2_row['year'])
                year_similarity = max(0, 1 - year_diff / 50)  # Normalize by 50 years
                similarity += year_similarity * 0.2
                factors += 0.2
        
        # Rating similarity
        if 'rating' in self.items_df.columns:
            if pd.notna(item1_row['rating']) and pd.notna(item2_row['rating']):
                rating_diff = abs(item1_row['rating'] - item2_row['rating'])
                rating_similarity = max(0, 1 - rating_diff / 5)  # Normalize by max rating difference
                similarity += rating_similarity * 0.3
                factors += 0.3
        
        # Duration similarity (for movies)
        if 'duration' in self.items_df.columns:
            if pd.notna(item1_row['duration']) and pd.notna(item2_row['duration']):
                # Extract numeric duration (assuming format like "142 min")
                try:
                    duration1 = float(str(item1_row['duration']).split()[0])
                    duration2 = float(str(item2_row['duration']).split()[0])
                    duration_diff = abs(duration1 - duration2)
                    duration_similarity = max(0, 1 - duration_diff / 120)  # Normalize by 120 minutes
                    similarity += duration_similarity * 0.1
                    factors += 0.1
                except (ValueError, IndexError):
                    pass
        
        # Normalize by total factors considered
        if factors > 0:
            similarity = similarity / factors
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def calculate_collaborative_score(self, target_item: str, user_ratings: Dict[str, int]) -> float:
        """
        Calculate collaborative filtering score for an item based on user ratings.
        
        Args:
            target_item: Item to score
            user_ratings: Dictionary of user ratings
            
        Returns:
            Collaborative score
        """
        if not user_ratings:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for rated_item, rating in user_ratings.items():
            if rated_item == target_item:
                continue
                
            similarity = self.calculate_content_similarity(target_item, rated_item)
            weight = similarity * abs(rating - 3)  # Weight by how much the rating deviates from neutral
            
            weighted_sum += (rating - 3) * weight  # Normalize ratings around neutral (3)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        collaborative_score = 3 + (weighted_sum / total_weight)  # Add back the neutral baseline
        return max(1, min(5, collaborative_score))  # Clamp to valid rating range
    
    def calculate_popularity_score(self, item_name: str) -> float:
        """
        Calculate popularity score based on average rating.
        
        Args:
            item_name: Name of the item
            
        Returns:
            Popularity score between 1 and 5
        """
        item_data = self.items_df[self.items_df['name'] == item_name]
        if item_data.empty or 'rating' not in self.items_df.columns:
            return 3.0  # Default neutral score
        
        rating = item_data.iloc[0]['rating']
        return rating if pd.notna(rating) else 3.0
    
    def generate_explanation(self, item_name: str, user_ratings: Dict[str, int], 
                           collab_score: float, popularity_score: float, final_score: float) -> str:
        """
        Generate human-readable explanation for the recommendation.
        
        Args:
            item_name: Recommended item name
            user_ratings: User's ratings
            collab_score: Collaborative filtering score
            popularity_score: Popularity score
            final_score: Final weighted score
            
        Returns:
            Explanation string
        """
        explanations = []
        
        # Find most similar liked items
        item_data = self.items_df[self.items_df['name'] == item_name].iloc[0]
        similar_items = []
        
        for rated_item, rating in user_ratings.items():
            if rating >= 4:  # User liked this item
                similarity = self.calculate_content_similarity(item_name, rated_item)
                if similarity > 0.3:  # Significant similarity
                    similar_items.append((rated_item, similarity, rating))
        
        similar_items.sort(key=lambda x: x[1], reverse=True)
        
        if similar_items:
            top_similar = similar_items[0]
            explanations.append(f"You rated '{top_similar[0]}' highly ({top_similar[2]}/5), and this item is {top_similar[1]:.0%} similar to it")
        
        # Genre explanation
        if 'genre' in item_data and pd.notna(item_data['genre']):
            genre_ratings = []
            for rated_item, rating in user_ratings.items():
                rated_data = self.items_df[self.items_df['name'] == rated_item]
                if not rated_data.empty and 'genre' in rated_data.columns:
                    rated_genre = rated_data.iloc[0]['genre']
                    if pd.notna(rated_genre) and rated_genre == item_data['genre']:
                        genre_ratings.append(rating)
            
            if genre_ratings:
                avg_genre_rating = np.mean(genre_ratings)
                if avg_genre_rating >= 4:
                    explanations.append(f"You seem to enjoy {item_data['genre']} items (avg rating: {avg_genre_rating:.1f}/5)")
        
        # Popularity explanation
        if popularity_score >= 4.5:
            explanations.append(f"This is a highly-rated item ({popularity_score:.1f}/5 average rating)")
        elif popularity_score >= 4.0:
            explanations.append(f"This item has good ratings ({popularity_score:.1f}/5 average rating)")
        
        # Confidence explanation
        if final_score >= 4.5:
            explanations.append("High confidence recommendation based on your preferences")
        elif final_score >= 4.0:
            explanations.append("Good match based on your rating patterns")
        else:
            explanations.append("Moderate recommendation - you might find this interesting")
        
        return ". ".join(explanations) + "."
    
    def get_recommendations(self, user_ratings: Dict[str, int], category: str, 
                          num_recommendations: int = 5, min_rating_threshold: int = 3) -> List[Tuple[str, float, str]]:
        """
        Generate recommendations for a user based on their ratings.
        
        Args:
            user_ratings: Dictionary mapping item names to ratings (1-5)
            category: Category to recommend from
            num_recommendations: Number of recommendations to return
            min_rating_threshold: Minimum user rating to consider as positive preference
            
        Returns:
            List of tuples (item_name, score, explanation)
        """
        if not user_ratings:
            return []
        
        # Filter items by category and exclude already rated items
        category_items = self.items_df[self.items_df['category'] == category]
        unrated_items = category_items[~category_items['name'].isin(list(user_ratings.keys()))]
        
        if len(unrated_items) == 0:
            return []
        
        recommendations = []
        
        # Filter user ratings by threshold
        positive_ratings = {k: v for k, v in user_ratings.items() if v >= min_rating_threshold}
        
        if not positive_ratings:
            # If no positive ratings, fall back to popularity-based recommendations
            for _, item in unrated_items.iterrows():
                item_name = str(item['name'])
                popularity_score = self.calculate_popularity_score(item_name)
                explanation = f"Popular item in {category} category (average rating: {popularity_score:.1f}/5). Rate some items to get personalized recommendations!"
                recommendations.append((item_name, popularity_score, explanation))
        else:
            # Generate recommendations based on collaborative filtering and popularity
            for _, item in unrated_items.iterrows():
                item_name = str(item['name'])
                
                # Calculate different components
                collaborative_score = self.calculate_collaborative_score(item_name, positive_ratings)
                popularity_score = self.calculate_popularity_score(item_name)
                
                # Weighted combination of different factors
                collab_weight = 0.7 if len(positive_ratings) >= 3 else 0.5  # More weight to collaborative if enough data
                popularity_weight = 1 - collab_weight
                
                final_score = (collaborative_score * collab_weight + 
                             popularity_score * popularity_weight)
                
                # Generate explanation
                explanation = self.generate_explanation(
                    item_name, positive_ratings, collaborative_score, popularity_score, final_score
                )
                
                recommendations.append((item_name, final_score, explanation))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:num_recommendations]
