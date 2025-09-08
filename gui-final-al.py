# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning for recommendations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Enhanced dictionary with default values
from collections import defaultdict

# GUI components
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading

# Date and time operations
from datetime import datetime

class UserManager:
    def __init__(self):
        # Store all user data
        self.users = {}
        # Track current logged in user
        self.current_user = None
        
    def register_user(self, username, password):
        # Check if username already exists
        if username in self.users:
            return False, "Username already exists!"
        
        # Create new user profile
        self.users[username] = {
            'password': password,
            'viewed': [],
            'rated': {},
            'history': [],
            'login_history': []
        }
        
        # Auto login new user
        self.current_user = username
        
        # Log registration time
        self.users[username]['login_history'].append({
            'action': 'registered',
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        return True, "Registration successful!"
        
    def login(self, username, password):
        # Get user data
        user = self.users.get(username)
        
        # Verify credentials
        if user and user['password'] == password:
            self.current_user = username
            
            # Log login time
            self.users[username]['login_history'].append({
                'action': 'login',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            return True, "Login successful!"
        return False, "Invalid credentials!"
    
    def logout(self):
        # Log logout time if user is logged in
        if self.current_user:
            self.users[self.current_user]['login_history'].append({
                'action': 'logout',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        # Clear current user
        self.current_user = None

class ResearchPaperRecommender:
    def __init__(self):
        # Main data storage - DataFrame containing all paper information
        self.papers = None
        # User management system for login/registration
        self.user_manager = UserManager()
        # TF-IDF matrix - numerical representation of text features for ML
        self.tfidf_matrix = None
        # Cosine similarity matrix - measures how similar papers are to each other
        self.similarity_matrix = None
        # TF-IDF vectorizer - converts text to numerical vectors for ML processing
        self.vectorizer = None
        # Initialize sample data and train the model
        self._create_sample_data()
        
    def _create_sample_data(self):
        """Create sample research papers dataset and train the recommendation model"""
        # Dictionary containing all paper data - structured format for DataFrame
        papers_data = {
            'paper_id': list(range(1, 21)),  # Unique IDs for each paper
            'title': [
                'Deep Learning Approaches to Computer Vision',
                'Natural Language Processing in Healthcare',
                'Reinforcement Learning for Robotics',
                'Transformer Models for Text Classification',
                'Medical Image Analysis Using CNNs',
                'Advanced Techniques in Neural Machine Translation',
                'AI-Driven Drug Discovery',
                'Computer Vision for Autonomous Vehicles',
                'BERT-based Clinical Text Mining',
                'Deep Reinforcement Learning in Game AI',
                'Quantum Computing for Machine Learning',
                'Blockchain Technology in Healthcare Systems',
                'Edge Computing for IoT Applications',
                'Federated Learning for Privacy-Preserving AI',
                'Graph Neural Networks for Social Network Analysis',
                'Explainable AI in Financial Decision Making',
                'Computer Vision for Agricultural Monitoring',
                'Natural Language Generation with GPT Models',
                'Time Series Forecasting with Deep Learning',
                'Cybersecurity Applications of Machine Learning'
            ],
            'abstract': [
                'Exploring modern DL techniques for image recognition tasks and their applications in various domains...',
                'Applying NLP methods to clinical text analysis for better healthcare outcomes...',
                'Developing RL algorithms for robotic control systems and autonomous navigation...',
                'Investigating transformer architectures for various NLP tasks and applications...',
                'CNN-based solutions for medical imaging diagnostics and disease detection...',
                'Improving NMT systems with attention mechanisms and modern architectures...',
                'Using AI for molecular structure analysis and accelerated drug discovery...',
                'Visual perception systems for self-driving cars and autonomous navigation...',
                'Mining clinical notes with BERT embeddings for healthcare insights...',
                'Applying DRL to complex game environments and strategic decision making...',
                'Leveraging quantum computing principles to enhance machine learning algorithms...',
                'Implementing blockchain for secure and transparent healthcare data management...',
                'Optimizing IoT device performance through edge computing architectures...',
                'Collaborative learning while preserving data privacy across distributed systems...',
                'Analyzing social networks using graph-based neural network approaches...',
                'Creating interpretable AI systems for transparent financial decisions...',
                'Using computer vision for crop monitoring and precision agriculture...',
                'Generating human-like text using advanced generative pre-trained models...',
                'Predicting future trends in time-dependent data using neural networks...',
                'Detecting and preventing cyber threats using machine learning techniques...'
            ],
            'keywords': [
                'deep learning, computer vision, image recognition',
                'nlp, healthcare, clinical text analysis',
                'reinforcement learning, robotics, control systems',
                'transformers, nlp, text classification',
                'medical imaging, cnn, diagnostics',
                'machine translation, attention mechanisms, neural networks',
                'drug discovery, ai, molecular analysis',
                'computer vision, autonomous vehicles, perception',
                'clinical text mining, bert, embeddings',
                'deep reinforcement learning, game ai',
                'quantum computing, machine learning, algorithms',
                'blockchain, healthcare, data security',
                'edge computing, iot, distributed systems',
                'federated learning, privacy, distributed ai',
                'graph neural networks, social networks, analysis',
                'explainable ai, finance, interpretability',
                'computer vision, agriculture, monitoring',
                'natural language generation, gpt, text generation',
                'time series, forecasting, deep learning',
                'cybersecurity, machine learning, threat detection'
            ],
            'category': [
                'Computer Vision', 'NLP', 'Reinforcement Learning',
                'NLP', 'Medical AI', 'NLP',
                'Medical AI', 'Computer Vision',
                'Medical AI', 'Reinforcement Learning',
                'Quantum Computing', 'Blockchain', 'Edge Computing',
                'Federated Learning', 'Graph Neural Networks', 'Explainable AI',
                'Computer Vision', 'NLP', 'Time Series', 'Cybersecurity'
            ],
            'average_rating': [4.2, 4.5, 3.8, 4.7, 4.1, 4.3, 3.9, 4.6, 4.4, 4.0,
                              3.7, 3.5, 4.1, 4.8, 3.9, 4.2, 3.6, 4.5, 4.3, 4.0],
            'total_ratings': [15, 22, 8, 31, 12, 18, 7, 25, 19, 13,
                             5, 4, 9, 27, 6, 14, 3, 23, 16, 11]
        }
        
        # Convert dictionary to pandas DataFrame for easier data manipulation
        self.papers = pd.DataFrame(papers_data)
        # Train the ML model using the paper data
        self._preprocess_data()
        
        # Add sample user data - Dictionary with user profiles and their interactions
        self.user_manager.users.update({
            'demo': {
                'password': 'demo123',
                'viewed': [1, 3, 5, 8, 12],  # List of paper IDs user has viewed
                'rated': {1: 5, 3: 4, 8: 5},  # Dictionary: paper_id -> rating
                'history': [],  # Search history
                'login_history': [  # List of login/logout activities with timestamps
                    {'action': 'registered', 'timestamp': '2024-01-15 10:30:45'},
                    {'action': 'login', 'timestamp': '2024-01-15 10:30:45'},
                    {'action': 'login', 'timestamp': '2024-01-16 14:22:13'},
                    {'action': 'logout', 'timestamp': '2024-01-16 15:45:32'}
                ]
            },
            'test': {
                'password': 'test123',
                'viewed': [2, 4, 6, 9, 15],
                'rated': {4: 5, 6: 4, 9: 3},
                'history': [],
                'login_history': [
                    {'action': 'registered', 'timestamp': '2024-01-14 09:15:22'},
                    {'action': 'login', 'timestamp': '2024-01-14 09:15:22'},
                    {'action': 'login', 'timestamp': '2024-01-17 11:33:18'}
                ]
            },
            'researcher': {
                'password': 'research123',
                'viewed': [7, 10, 11, 14, 16, 18],
                'rated': {7: 4, 10: 5, 14: 5, 16: 3},
                'history': [],
                'login_history': [
                    {'action': 'registered', 'timestamp': '2024-01-10 16:45:33'},
                    {'action': 'login', 'timestamp': '2024-01-10 16:45:33'},
                    {'action': 'login', 'timestamp': '2024-01-12 08:20:15'},
                    {'action': 'logout', 'timestamp': '2024-01-12 17:30:22'},
                    {'action': 'login', 'timestamp': '2024-01-18 13:45:10'}
                ]
            }
        })
        
    def _preprocess_data(self):
        """Train the machine learning model for recommendations"""
        # Combine all text features into one column for ML processing
        self.papers['combined_features'] = (
            self.papers['title'] + ' ' +
            self.papers['abstract'] + ' ' +
            self.papers['keywords'] + ' ' +
            self.papers['category']
        )
        
        # Initialize TF-IDF vectorizer - converts text to numerical features
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Training: Transform text data into numerical matrix for ML algorithms
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.papers['combined_features']
        )
        
        # Calculate similarity between all papers - creates similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
    def content_based_recommendations(self, paper_id, top_n=5):
        """Find similar papers based on content similarity (ML-based)"""
        # Find paper index in DataFrame
        idx = self.papers.index[self.papers['paper_id'] == paper_id].tolist()[0]
        # Get similarity scores for this paper with all other papers
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        # Sort by similarity score (highest first)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get top N similar papers (excluding the paper itself)
        sim_scores = sim_scores[1:top_n+1]
        # Extract paper indices
        paper_indices = [i[0] for i in sim_scores]
        # Return recommended papers
        return self.papers.iloc[paper_indices]
    
    def user_based_recommendations(self, top_n=5):
        """Generate personalized recommendations based on user behavior"""
        # Check if user is logged in
        if not self.user_manager.current_user:
            return []
            
        # Get current user's data dictionary
        user_data = self.user_manager.users[self.user_manager.current_user]
        rated_papers = user_data['rated']  # Dictionary of user's ratings
        
        # If user has no history, return popular papers
        if not rated_papers and not user_data['viewed']:
            return self._get_popular_papers(top_n)
        
        # Dictionary to accumulate recommendation scores
        all_scores = defaultdict(float)
        
        # Generate recommendations based on rated papers
        for pid, rating in rated_papers.items():
            # Find papers similar to this rated paper
            similar_papers = self.content_based_recommendations(pid, 10)
            for _, paper in similar_papers.iterrows():
                # Only recommend unrated papers
                if paper['paper_id'] not in rated_papers:
                    # Weight recommendation by user's rating and similarity score
                    all_scores[paper['paper_id']] += rating * \
                        self.similarity_matrix[pid-1][paper['paper_id']-1]
        
        # Add recommendations based on viewed papers (lower weight)
        for pid in user_data['viewed']:
            similar_papers = self.content_based_recommendations(pid, 5)
            for _, paper in similar_papers.iterrows():
                # Weight viewed papers less than rated papers
                all_scores[paper['paper_id']] += 0.5 * \
                    self.similarity_matrix[pid-1][paper['paper_id']-1]
        
        # Exclude papers user has already seen
        seen_papers = set(user_data['viewed']).union(set(rated_papers.keys()))
        # Sort recommendations by score and return top N
        recommendations = sorted(
            [(pid, score) for pid, score in all_scores.items() if pid not in seen_papers],
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Return list of recommended paper IDs
        return [pid for pid, _ in recommendations]
    
    def _get_popular_papers(self, top_n):
        """Get most popular papers based on ratings"""
        # Sort by average rating and total ratings
        popular_papers = self.papers.sort_values(
            ['average_rating', 'total_ratings'], 
            ascending=[False, False]
        ).head(top_n)
        return popular_papers['paper_id'].tolist()
    
    def search_papers(self, query, top_n=5):
        """Search papers using ML-based text similarity"""
        # Convert search query to numerical vector using trained vectorizer
        query_vec = self.vectorizer.transform([query])
        # Calculate similarity between query and all papers
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        # Get indices of most similar papers
        top_indices = sim_scores.argsort()[-top_n:][::-1]
        # Get search results
        results = self.papers.iloc[top_indices]
        
        # Add to user's search history if logged in
        if self.user_manager.current_user:
            self.user_manager.users[self.user_manager.current_user]['history'].extend(
                results['paper_id'].tolist()
            )
        return results
    
    def rate_paper(self, paper_id, rating):
        """Allow user to rate a paper and update system data"""
        # Validation checks
        if not self.user_manager.current_user:
            return False, "Please login first"
            
        if paper_id not in self.papers['paper_id'].values:
            return False, "Invalid paper ID"
            
        if rating < 1 or rating > 5:
            return False, "Rating must be between 1-5"
            
        # Get user's data dictionary
        user_data = self.user_manager.users[self.user_manager.current_user]
        old_rating = user_data['rated'].get(paper_id, 0)
        # Update user's rating dictionary
        user_data['rated'][paper_id] = rating
        
        # Update paper's average rating in the main dataset
        paper_idx = self.papers[self.papers['paper_id'] == paper_id].index[0]
        current_avg = self.papers.loc[paper_idx, 'average_rating']
        current_total = self.papers.loc[paper_idx, 'total_ratings']
        
        if old_rating == 0:  # New rating - update totals
            new_avg = (current_avg * current_total + rating) / (current_total + 1)
            self.papers.loc[paper_idx, 'average_rating'] = round(new_avg, 1)
            self.papers.loc[paper_idx, 'total_ratings'] = current_total + 1
        else:  # Update existing rating - maintain same total count
            new_avg = (current_avg * current_total - old_rating + rating) / current_total
            self.papers.loc[paper_idx, 'average_rating'] = round(new_avg, 1)
        
        return True, "Rating submitted successfully!"
    
    def get_user_ratings(self):
        """Get current user's ratings dictionary"""
        if not self.user_manager.current_user:
            return {}
        return self.user_manager.users[self.user_manager.current_user]['rated']

    def get_paper_statistics(self):
        """Get comprehensive statistics about the paper dataset"""
        # Dictionary containing various statistics
        stats = {
            'total_papers': len(self.papers),
            # Dictionary showing count of papers per category
            'categories': self.papers['category'].value_counts().to_dict(),
            # Statistical description of rating distribution
            'avg_rating_distribution': self.papers['average_rating'].describe(),
            # List of dictionaries for top rated papers
            'top_rated': self.papers.nlargest(3, 'average_rating')[['title', 'average_rating']].to_dict('records'),
            # List of dictionaries for most reviewed papers
            'most_rated': self.papers.nlargest(3, 'total_ratings')[['title', 'total_ratings']].to_dict('records')
        }
        return stats

class PaperRecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Research Paper Recommendation System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.recommender = ResearchPaperRecommender()
        self.current_results = None
        
        self.setup_styles()
        self.create_widgets()
        self.show_login_page()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Custom.TButton', font=('Arial', 10))
        
    def create_widgets(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Enhanced Research Paper Recommendation System", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel for controls
        self.left_panel = ttk.Frame(self.main_frame, width=350)
        self.left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        self.left_panel.grid_propagate(False)
        
        # Right panel for results
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=1)
        
        self.create_left_panel()
        self.create_right_panel()
        
    def create_left_panel(self):
        # Login/User section
        self.user_frame = ttk.LabelFrame(self.left_panel, text="User Account", padding="10")
        self.user_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.left_panel.columnconfigure(0, weight=1)
        
        # Search section
        self.search_frame = ttk.LabelFrame(self.left_panel, text="Search Papers", padding="10")
        self.search_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Actions section
        self.actions_frame = ttk.LabelFrame(self.left_panel, text="Actions", padding="10")
        self.actions_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Rating section
        self.rating_frame = ttk.LabelFrame(self.left_panel, text="Rate Paper", padding="10")
        self.rating_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.setup_user_section()
        self.setup_search_section()
        self.setup_actions_section()
        self.setup_rating_section()
        
    def setup_user_section(self):
        # User status
        self.user_status = ttk.Label(self.user_frame, text="Not logged in")
        self.user_status.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Login fields
        ttk.Label(self.user_frame, text="Username:").grid(row=1, column=0, sticky=tk.W)
        self.username_entry = ttk.Entry(self.user_frame, width=20)
        self.username_entry.grid(row=1, column=1, pady=2)
        
        ttk.Label(self.user_frame, text="Password:").grid(row=2, column=0, sticky=tk.W)
        self.password_entry = ttk.Entry(self.user_frame, show="*", width=20)
        self.password_entry.grid(row=2, column=1, pady=2)
        
        # Buttons
        self.login_btn = ttk.Button(self.user_frame, text="Login", command=self.login)
        self.login_btn.grid(row=3, column=0, pady=5, sticky=tk.W)
        
        self.register_btn = ttk.Button(self.user_frame, text="Register", command=self.register)
        self.register_btn.grid(row=3, column=1, pady=5, sticky=tk.E)
        
        self.logout_btn = ttk.Button(self.user_frame, text="Logout", command=self.logout, state='disabled')
        self.logout_btn.grid(row=4, column=0, columnspan=2, pady=5)
        
    def setup_search_section(self):
        ttk.Label(self.search_frame, text="Search Query:").grid(row=0, column=0, sticky=tk.W)
        self.search_entry = ttk.Entry(self.search_frame, width=25)
        self.search_entry.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        self.search_entry.bind('<Return>', lambda e: self.search_papers())
        
        self.search_btn = ttk.Button(self.search_frame, text="Search", command=self.search_papers)
        self.search_btn.grid(row=2, column=0, pady=5)
        
        self.search_frame.columnconfigure(0, weight=1)
        
    def setup_actions_section(self):
        self.recommend_btn = ttk.Button(self.actions_frame, text="Get Recommendations", 
                                       command=self.get_recommendations, state='disabled')
        self.recommend_btn.grid(row=0, column=0, pady=3, sticky=(tk.W, tk.E))
        
        self.history_btn = ttk.Button(self.actions_frame, text="View History", 
                                     command=self.view_history, state='disabled')
        self.history_btn.grid(row=1, column=0, pady=3, sticky=(tk.W, tk.E))
        
        self.ratings_btn = ttk.Button(self.actions_frame, text="My Ratings", 
                                     command=self.view_user_ratings, state='disabled')
        self.ratings_btn.grid(row=2, column=0, pady=3, sticky=(tk.W, tk.E))
        
        self.login_history_btn = ttk.Button(self.actions_frame, text="Login History", 
                                           command=self.view_login_history, state='disabled')
        self.login_history_btn.grid(row=3, column=0, pady=3, sticky=(tk.W, tk.E))
        
        self.all_papers_btn = ttk.Button(self.actions_frame, text="Show All Papers", 
                                        command=self.show_all_papers)
        self.all_papers_btn.grid(row=4, column=0, pady=3, sticky=(tk.W, tk.E))
        
        # New statistics button
        self.stats_btn = ttk.Button(self.actions_frame, text="System Statistics", 
                                   command=self.show_statistics)
        self.stats_btn.grid(row=5, column=0, pady=3, sticky=(tk.W, tk.E))
        
        self.actions_frame.columnconfigure(0, weight=1)
        
    def setup_rating_section(self):
        ttk.Label(self.rating_frame, text="Paper ID:").grid(row=0, column=0, sticky=tk.W)
        self.paper_id_entry = ttk.Entry(self.rating_frame, width=10)
        self.paper_id_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(self.rating_frame, text="Rating (1-5):").grid(row=1, column=0, sticky=tk.W)
        self.rating_var = tk.StringVar()
        self.rating_combo = ttk.Combobox(self.rating_frame, textvariable=self.rating_var, 
                                        values=['1', '2', '3', '4', '5'], width=8, state='readonly')
        self.rating_combo.grid(row=1, column=1, padx=5, pady=5)
        
        self.rate_btn = ttk.Button(self.rating_frame, text="Submit Rating", 
                                  command=self.rate_paper, state='disabled')
        self.rate_btn.grid(row=2, column=0, columnspan=2, pady=5)
        
    def create_right_panel(self):
        # Results display
        results_label = ttk.Label(self.right_panel, text="Results", style='Header.TLabel')
        results_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Scrolled text for results
        self.results_text = scrolledtext.ScrolledText(self.right_panel, width=70, height=40, 
                                                     font=('Consolas', 9))
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def show_login_page(self):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Welcome to Enhanced Research Paper Recommendation System!\n\n")
        self.results_text.insert(tk.END, "Please login or register to get started.\n\n")
        self.results_text.insert(tk.END, "Demo accounts:\n")
        self.results_text.insert(tk.END, "Username: demo, Password: demo123\n")
        self.results_text.insert(tk.END, "Username: test, Password: test123\n")
        self.results_text.insert(tk.END, "Username: researcher, Password: research123\n\n")
        self.results_text.insert(tk.END, "New Features:\n")
        self.results_text.insert(tk.END, "• View paper ratings and statistics\n")
        self.results_text.insert(tk.END, "• Enhanced paper database (20 papers)\n")
        self.results_text.insert(tk.END, "• User login history tracking\n")
        self.results_text.insert(tk.END, "• Personal rating history\n")
        self.results_text.insert(tk.END, "• Improved recommendation algorithm\n")
        self.results_text.insert(tk.END, "• System-wide statistics dashboard\n")
                
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
            
        success, message = self.recommender.user_manager.login(username, password)
        
        if success:
            self.update_login_state(True)
            messagebox.showinfo("Success", message)
            self.show_welcome_message()
        else:
            messagebox.showerror("Error", message)
            
    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
            
        success, message = self.recommender.user_manager.register_user(username, password)
        
        if success:
            self.update_login_state(True)
            messagebox.showinfo("Success", message)
            self.show_welcome_message()
        else:
            messagebox.showerror("Error", message)
            
    def logout(self):
        self.recommender.user_manager.logout()
        self.update_login_state(False)
        self.show_login_page()
        
    def update_login_state(self, logged_in):
        if logged_in:
            user = self.recommender.user_manager.current_user
            self.user_status.config(text=f"Logged in as: {user}")
            self.login_btn.config(state='disabled')
            self.register_btn.config(state='disabled')
            self.logout_btn.config(state='normal')
            self.recommend_btn.config(state='normal')
            self.history_btn.config(state='normal')
            self.rate_btn.config(state='normal')
            self.ratings_btn.config(state='normal')
            self.login_history_btn.config(state='normal')
        else:
            self.user_status.config(text="Not logged in")
            self.login_btn.config(state='normal')
            self.register_btn.config(state='normal')
            self.logout_btn.config(state='disabled')
            self.recommend_btn.config(state='disabled')
            self.history_btn.config(state='disabled')
            self.rate_btn.config(state='disabled')
            self.ratings_btn.config(state='disabled')
            self.login_history_btn.config(state='disabled')
    
    def show_welcome_message(self):
        """Show personalized welcome message after login"""
        user = self.recommender.user_manager.current_user
        user_data = self.recommender.user_manager.users[user]
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Welcome back, {user}!\n\n")
        self.results_text.insert(tk.END, f"Your Activity Summary:\n")
        self.results_text.insert(tk.END, f"• Papers viewed: {len(user_data['viewed'])}\n")
        self.results_text.insert(tk.END, f"• Papers rated: {len(user_data['rated'])}\n")
        self.results_text.insert(tk.END, f"• Login sessions: {len(user_data['login_history'])}\n\n")
        
        if user_data['rated']:
            avg_rating = sum(user_data['rated'].values()) / len(user_data['rated'])
            self.results_text.insert(tk.END, f"Your average rating: {avg_rating:.1f}/5\n\n")
        
        self.results_text.insert(tk.END, "What would you like to do?\n")
        self.results_text.insert(tk.END, "• Search for papers\n")
        self.results_text.insert(tk.END, "• Get personalized recommendations\n")
        self.results_text.insert(tk.END, "• View your activity history\n")
        self.results_text.insert(tk.END, "• Rate papers you've read\n")
            
    def search_papers(self):
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showerror("Error", "Please enter a search query")
            return
            
        results = self.recommender.search_papers(query, 5)
        self.display_results(results, f"Search Results for: '{query}'")
        
        # Mark papers as viewed
        if self.recommender.user_manager.current_user:
            user_data = self.recommender.user_manager.users[self.recommender.user_manager.current_user]
            viewed_ids = results['paper_id'].tolist()
            user_data['viewed'].extend(viewed_ids)
            user_data['viewed'] = list(set(user_data['viewed']))  # Remove duplicates
            
    def get_recommendations(self):
        if not self.recommender.user_manager.current_user:
            messagebox.showerror("Error", "Please login first")
            return
            
        rec_ids = self.recommender.user_based_recommendations(5)
        
        if not rec_ids:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No recommendations available.\n\n")
            self.results_text.insert(tk.END, "Try rating some papers or searching first!")
            return
            
        recommendations = self.recommender.papers[self.recommender.papers['paper_id'].isin(rec_ids)]
        self.display_results(recommendations, "Personalized Recommendations")
        
    def view_history(self):
        if not self.recommender.user_manager.current_user:
            messagebox.showerror("Error", "Please login first")
            return
            
        user_data = self.recommender.user_manager.users[self.recommender.user_manager.current_user]
        viewed_papers = self.recommender.papers[self.recommender.papers['paper_id'].isin(user_data['viewed'])]
        
        if viewed_papers.empty:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No viewing history found.\n\n")
            self.results_text.insert(tk.END, "Start by searching for papers!")
            return
            
        self.display_results(viewed_papers, "Your Viewing History")
    
    def view_user_ratings(self):
        if not self.recommender.user_manager.current_user:
            messagebox.showerror("Error", "Please login first")
            return
            
        user_ratings = self.recommender.get_user_ratings()
        
        if not user_ratings:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "You haven't rated any papers yet.\n\n")
            self.results_text.insert(tk.END, "Rate some papers to see your rating history!")
            return
        
        # Get rated papers details
        rated_paper_ids = list(user_ratings.keys())
        rated_papers = self.recommender.papers[self.recommender.papers['paper_id'].isin(rated_paper_ids)]
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Your Paper Ratings\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        total_ratings = len(user_ratings)
        avg_rating = sum(user_ratings.values()) / total_ratings
        self.results_text.insert(tk.END, f"Total papers rated: {total_ratings}\n")
        self.results_text.insert(tk.END, f"Your average rating: {avg_rating:.1f}/5\n\n")
        
        # Sort by rating (highest first)
        sorted_ratings = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
        
        for paper_id, rating in sorted_ratings:
            paper = rated_papers[rated_papers['paper_id'] == paper_id].iloc[0]
            stars = "★" * rating + "☆" * (5 - rating)
            
            self.results_text.insert(tk.END, f"ID: {paper_id} | Rating: {rating}/5 {stars}\n")
            self.results_text.insert(tk.END, f"Title: {paper['title']}\n")
            self.results_text.insert(tk.END, f"Category: {paper['category']}\n")
            self.results_text.insert(tk.END, f"Current avg rating: {paper['average_rating']}/5 ({paper['total_ratings']} ratings)\n")
            self.results_text.insert(tk.END, "-" * 70 + "\n\n")
    
    def view_login_history(self):
        if not self.recommender.user_manager.current_user:
            messagebox.showerror("Error", "Please login first")
            return
            
        user = self.recommender.user_manager.current_user
        login_history = self.recommender.user_manager.users[user]['login_history']
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Login History for {user}\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        if not login_history:
            self.results_text.insert(tk.END, "No login history available.")
            return
        
        # Show recent activities first
        for entry in reversed(login_history):
            action = entry['action'].title()
            timestamp = entry['timestamp']
            
            if action == 'Login':
                icon = "🟢"
            elif action == 'Logout':
                icon = "🔴"
            else:  # Registration
                icon = "🆕"
            
            self.results_text.insert(tk.END, f"{icon} {action}: {timestamp}\n")
        
        # Show statistics
        logins = len([e for e in login_history if e['action'] == 'login'])
        logouts = len([e for e in login_history if e['action'] == 'logout'])
        
        self.results_text.insert(tk.END, f"\n--- Statistics ---\n")
        self.results_text.insert(tk.END, f"Total logins: {logins}\n")
        self.results_text.insert(tk.END, f"Total logouts: {logouts}\n")
        self.results_text.insert(tk.END, f"Currently logged in: {'Yes' if logins > logouts else 'No'}\n")
    
    def show_all_papers(self):
        self.display_results(self.recommender.papers, "All Research Papers")
    
    def show_statistics(self):
        stats = self.recommender.get_paper_statistics()
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "System Statistics Dashboard\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # General statistics
        self.results_text.insert(tk.END, f"📊 General Statistics\n")
        self.results_text.insert(tk.END, f"Total papers in database: {stats['total_papers']}\n")
        self.results_text.insert(tk.END, f"Total registered users: {len(self.recommender.user_manager.users)}\n\n")
        
        # Category distribution
        self.results_text.insert(tk.END, f"📈 Papers by Category\n")
        for category, count in stats['categories'].items():
            percentage = (count / stats['total_papers']) * 100
            self.results_text.insert(tk.END, f"• {category}: {count} papers ({percentage:.1f}%)\n")
        
        # Rating statistics
        self.results_text.insert(tk.END, f"\n⭐ Rating Statistics\n")
        rating_stats = stats['avg_rating_distribution']
        self.results_text.insert(tk.END, f"Average rating across all papers: {rating_stats['mean']:.2f}/5\n")
        self.results_text.insert(tk.END, f"Highest rated paper: {rating_stats['max']:.1f}/5\n")
        self.results_text.insert(tk.END, f"Lowest rated paper: {rating_stats['min']:.1f}/5\n")
        self.results_text.insert(tk.END, f"Standard deviation: {rating_stats['std']:.2f}\n\n")
        
        # Top rated papers
        self.results_text.insert(tk.END, f"🏆 Top Rated Papers\n")
        for i, paper in enumerate(stats['top_rated'], 1):
            self.results_text.insert(tk.END, f"{i}. {paper['title']} ({paper['average_rating']}/5)\n")
        
        # Most rated papers
        self.results_text.insert(tk.END, f"\n🔥 Most Rated Papers\n")
        for i, paper in enumerate(stats['most_rated'], 1):
            self.results_text.insert(tk.END, f"{i}. {paper['title']} ({paper['total_ratings']} ratings)\n")
        
        # User activity statistics
        if self.recommender.user_manager.current_user:
            user_data = self.recommender.user_manager.users[self.recommender.user_manager.current_user]
            self.results_text.insert(tk.END, f"\n👤 Your Personal Stats\n")
            self.results_text.insert(tk.END, f"Papers viewed: {len(user_data['viewed'])}\n")
            self.results_text.insert(tk.END, f"Papers rated: {len(user_data['rated'])}\n")
            if user_data['rated']:
                user_avg = sum(user_data['rated'].values()) / len(user_data['rated'])
                self.results_text.insert(tk.END, f"Your average rating: {user_avg:.1f}/5\n")
    
    def rate_paper(self):
        if not self.recommender.user_manager.current_user:
            messagebox.showerror("Error", "Please login first")
            return
            
        try:
            paper_id = int(self.paper_id_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid paper ID (number)")
            return
            
        rating_str = self.rating_var.get()
        if not rating_str:
            messagebox.showerror("Error", "Please select a rating")
            return
            
        rating = int(rating_str)
        success, message = self.recommender.rate_paper(paper_id, rating)
        
        if success:
            messagebox.showinfo("Success", message)
            self.paper_id_entry.delete(0, tk.END)
            self.rating_var.set("")
            
            # Show the rated paper details
            paper = self.recommender.papers[self.recommender.papers['paper_id'] == paper_id]
            if not paper.empty:
                self.display_results(paper, f"Paper #{paper_id} - Your Rating: {rating}/5")
        else:
            messagebox.showerror("Error", message)
    
    def display_results(self, papers_df, title):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"{title}\n")
        self.results_text.insert(tk.END, "=" * len(title) + "\n\n")
        
        if papers_df.empty:
            self.results_text.insert(tk.END, "No papers found.")
            return
        
        user_ratings = {}
        if self.recommender.user_manager.current_user:
            user_ratings = self.recommender.get_user_ratings()
        
        for idx, paper in papers_df.iterrows():
            # Paper header with rating info
            stars = "★" * int(paper['average_rating']) + "☆" * (5 - int(paper['average_rating']))
            user_rating_text = ""
            if paper['paper_id'] in user_ratings:
                user_stars = "★" * user_ratings[paper['paper_id']] + "☆" * (5 - user_ratings[paper['paper_id']])
                user_rating_text = f" | Your rating: {user_ratings[paper['paper_id']]}/5 {user_stars}"
            
            self.results_text.insert(tk.END, f"📄 Paper ID: {paper['paper_id']}\n")
            self.results_text.insert(tk.END, f"Title: {paper['title']}\n")
            self.results_text.insert(tk.END, f"Category: {paper['category']}\n")
            self.results_text.insert(tk.END, f"Rating: {paper['average_rating']}/5 {stars} ({paper['total_ratings']} ratings){user_rating_text}\n")
            self.results_text.insert(tk.END, f"Keywords: {paper['keywords']}\n")
            self.results_text.insert(tk.END, f"Abstract: {paper['abstract']}\n")
            self.results_text.insert(tk.END, "-" * 80 + "\n\n")
        
        # Show summary
        total_papers = len(papers_df)
        avg_rating = papers_df['average_rating'].mean()
        self.results_text.insert(tk.END, f"📊 Summary: {total_papers} papers shown | Average rating: {avg_rating:.2f}/5\n")

def main():
    root = tk.Tk()
    app = PaperRecommenderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()