# Cybersecurity ML Framework - Web Interface

## 🌐 **Web Frontend for Cybersecurity ML Framework**

A beautiful, interactive web interface for your cybersecurity machine learning framework built with Flask and Bootstrap.

## 🚀 **Quick Start**

### **Method 1: Simple Start**
```bash
cd "/Users/prabhathbhuma/Desktop/Report writing"
source venv/bin/activate
python start_web_interface.py
```

### **Method 2: Direct Start**
```bash
cd "/Users/prabhathbhuma/Desktop/Report writing/frontend"
source ../venv/bin/activate
python app.py
```

### **Method 3: Test First**
```bash
cd "/Users/prabhathbhuma/Desktop/Report writing"
source venv/bin/activate
python test_web_interface.py
```

## 🌐 **Access the Interface**

Once started, open your web browser and go to:
**http://localhost:5000**

## ✨ **Features**

### **1. Data Generation**
- Generate synthetic cybersecurity datasets
- Configurable parameters (samples, features, classes)
- Real-time dataset information display

### **2. Model Training**
- **Random Forest**: High-performance ensemble classifier
- **Support Vector Machine**: Robust classification with kernel methods
- Configurable hyperparameters
- Real-time training progress

### **3. Anomaly Detection**
- **Isolation Forest**: Unsupervised anomaly detection
- **One-Class SVM**: Support vector-based detection
- Configurable contamination levels
- Anomaly detection statistics

### **4. Prediction**
- Make predictions on new data
- Input validation and error handling
- Confidence scores and probabilities
- Sample data generation

### **5. Visualizations**
- **Confusion Matrix**: Model performance visualization
- **Feature Importance**: Understanding key features
- **ROC Curve**: Classification performance
- Interactive plot generation

### **6. Real-time Monitoring**
- System status dashboard
- Model and dataset counters
- Live updates every 5 seconds
- Error handling and notifications

## 🎨 **Interface Design**

### **Modern UI Features**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Bootstrap 5**: Modern, clean interface
- **Gradient Backgrounds**: Professional cybersecurity theme
- **Interactive Cards**: Hover effects and animations
- **Real-time Updates**: Live status monitoring
- **Alert System**: Success, warning, and error notifications

### **Color Scheme**
- **Primary**: Blue gradient (#667eea to #764ba2)
- **Success**: Green gradient (#56ab2f to #a8e6cf)
- **Warning**: Pink gradient (#f093fb to #f5576c)
- **Info**: Cyan gradient (#4facfe to #00f2fe)

## 📊 **Usage Guide**

### **Step 1: Generate Dataset**
1. Set parameters (samples, features, classes)
2. Click "Generate Dataset"
3. Wait for confirmation

### **Step 2: Train Models**
1. Select model type (Random Forest or SVM)
2. Configure parameters
3. Click "Train Model"
4. View accuracy results

### **Step 3: Train Anomaly Detectors**
1. Select detector type
2. Set contamination level
3. Click "Train Detector"
4. View detection statistics

### **Step 4: Make Predictions**
1. Select trained model
2. Enter feature values (comma-separated)
3. Click "Predict"
4. View prediction and confidence

### **Step 5: Generate Visualizations**
1. Select plot type
2. Choose trained model
3. Click "Generate Plot"
4. View interactive visualization

## 🔧 **Technical Details**

### **Backend (Flask)**
- **Framework**: Flask 3.1.2
- **API Endpoints**: RESTful JSON API
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Error Handling**: Comprehensive exception handling

### **Frontend (HTML/CSS/JS)**
- **Framework**: Bootstrap 5.1.3
- **Icons**: Font Awesome 6.0.0
- **JavaScript**: Vanilla JS with fetch API
- **Styling**: Custom CSS with gradients and animations
- **Responsiveness**: Mobile-first design

### **API Endpoints**
```
GET  /api/get_status              - System status
POST /api/generate_data           - Generate dataset
POST /api/train_model             - Train ML model
POST /api/train_anomaly_detector  - Train anomaly detector
POST /api/predict                 - Make prediction
POST /api/visualize               - Generate visualization
```

## 📁 **File Structure**

```
frontend/
├── app.py                    # Flask application
├── templates/
│   └── index.html           # Main HTML template
├── static/
│   ├── css/                 # Custom CSS (if any)
│   ├── js/
│   │   └── app.js          # Frontend JavaScript
│   └── images/             # Static images
├── start_web_interface.py  # Startup script
└── test_web_interface.py   # Test script
```

## 🧪 **Testing**

### **Automated Test**
```bash
python test_web_interface.py
```

**Test Coverage:**
- ✅ Server connectivity
- ✅ Data generation
- ✅ Model training
- ✅ Prediction making
- ✅ Visualization generation

### **Manual Testing**
1. Open http://localhost:5000
2. Generate a dataset
3. Train a model
4. Make predictions
5. Generate visualizations

## 🐛 **Troubleshooting**

### **Common Issues**

**1. Server won't start**
```bash
# Check if port 5000 is available
lsof -i :5000

# Kill existing process
kill -9 $(lsof -t -i:5000)

# Restart server
python start_web_interface.py
```

**2. Import errors**
```bash
# Install missing packages
pip install flask numpy pandas scikit-learn matplotlib seaborn requests
```

**3. Permission errors**
```bash
# Make scripts executable
chmod +x start_web_interface.py test_web_interface.py
```

**4. Browser issues**
- Clear browser cache
- Try different browser
- Check JavaScript console for errors

### **Debug Mode**
The Flask app runs in debug mode by default, which provides:
- Automatic reloading on code changes
- Detailed error messages
- Interactive debugger

## 🔒 **Security Features**

- **Input Validation**: All user inputs are validated
- **Error Handling**: Comprehensive error catching
- **CORS Protection**: Configured for local development
- **Data Sanitization**: Prevents injection attacks

## 📈 **Performance**

- **Fast Loading**: Optimized static assets
- **Efficient API**: Minimal data transfer
- **Caching**: Browser caching for static files
- **Responsive**: Smooth animations and transitions

## 🎯 **Future Enhancements**

- **User Authentication**: Login system
- **Data Upload**: CSV file upload
- **Model Persistence**: Save/load trained models
- **Real-time Monitoring**: Live data streaming
- **Advanced Visualizations**: Interactive charts
- **Export Features**: Download results and plots

## 🌟 **Screenshots**

The interface includes:
- **Dashboard**: System status and metrics
- **Data Generation**: Parameter configuration
- **Model Training**: Real-time progress
- **Prediction Interface**: Input validation
- **Visualization Gallery**: Interactive plots
- **Responsive Design**: Mobile-friendly layout

## 🎉 **Success!**

Your cybersecurity ML framework now has a beautiful, functional web interface! 

**Features Working:**
- ✅ Data generation and management
- ✅ ML model training and evaluation
- ✅ Anomaly detection
- ✅ Real-time predictions
- ✅ Interactive visualizations
- ✅ Responsive design
- ✅ Error handling
- ✅ Status monitoring

**Next Steps:**
1. Open http://localhost:5000 in your browser
2. Generate a dataset
3. Train models
4. Make predictions
5. Generate visualizations
6. Explore all features!

---

**🎊 Congratulations! Your cybersecurity ML framework now has a complete web interface!**
