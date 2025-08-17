import React, { useState, useRef } from 'react';
import './VQAInterface.css';

function VQAInterface() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelType, setModelType] = useState('radiology-open');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.type.startsWith('image/')) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
      // Reset answer when a new image is uploaded
      setAnswer(null);
      setError(null);
    } else {
      setError('Please select an image file');
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleModelChange = (e) => {
    setModelType(e.target.value);
    // Reset answer when model is changed
    setAnswer(null);
  };

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setIsProcessing(true);
    setError(null);
    
    // Create form data for file upload
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', modelType);
    formData.append('question', question);

    try {
      const response = await fetch('/api/vqa', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setAnswer(data.answer);
        } else {
          setError(data.message || 'Failed to process request');
        }
      } else {
        setError('Server error, please try again');
      }
    } catch (error) {
      console.error('Error processing image:', error);
      setError('Network error, please check your connection');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnswer(null);
    setError(null);
  };

  const getModelName = (type) => {
    switch(type) {
      case 'radiology':
        return 'Radiology';
      case 'pathology':
        return 'Pathology';
      default:
        return 'Unknown Model';
    }
  };

  return (
    <div className="vqa-container">
      <div className="model-selection">
        <h3>Select Model</h3>
        <div className="model-options">
          <div className="model-option">
            <input
              type="radio"
              id="radiology"
              name="model"
              value="radiology"
              checked={modelType === 'radiology'}
              onChange={handleModelChange}
            />
            <label htmlFor="radiology-open">Radiology</label>
          </div>
           
          <div className="model-option">
            <input
              type="radio"
              id="pathology-open"
              name="model"
              value="pathology"
              checked={modelType === 'pathology'}
              onChange={handleModelChange}
            />
            <label htmlFor="pathology-open">Pathology</label>
          </div>
        </div>
      </div>

      <div className="vqa-interface-wrapper">
        <div className="image-upload-section">
          <h3>Upload Image</h3>
          <div 
            className={`dropzone ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-preview' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={!previewUrl ? triggerFileInput : undefined}
          >
            <input 
              type="file" 
              ref={fileInputRef}
              onChange={handleFileInput}
              accept="image/*"
              className="file-input"
            />
            
            {previewUrl ? (
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="image-preview" />
                <button type="button" className="clear-button" onClick={clearImage}>
                  &times;
                </button>
              </div>
            ) : (
              <div className="dropzone-content">
                <div className="upload-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="17 8 12 3 7 8"></polyline>
                    <line x1="12" y1="3" x2="12" y2="15"></line>
                  </svg>
                </div>
                <p className="dropzone-text">Drag & Drop your {modelType.includes('radiology') ? 'radiology' : 'pathology'} image here or click to browse</p>
                <p className="dropzone-hint">Supports: JPG, PNG, DICOM</p>
              </div>
            )}
          </div>
        </div>

        <div className="question-section">
          <h3>Ask Question</h3>
          <div className="question-input-container">
            <textarea
              className="question-input"
              placeholder={`Ask a ${modelType.includes('closed') ? 'yes/no' : 'descriptive'} question about the ${modelType.includes('radiology') ? 'radiology' : 'pathology'} image...`}
              value={question}
              onChange={handleQuestionChange}
              disabled={!selectedFile}
            ></textarea>
            <button 
              className={`submit-button ${!selectedFile || !question.trim() || isProcessing ? 'disabled' : ''}`}
              onClick={handleSubmit}
              disabled={!selectedFile || !question.trim() || isProcessing}
            >
              {isProcessing ? 'Processing...' : 'Get Answer'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {answer && (
        <div className="result-section">
          <h3>Result from {getModelName(modelType)}</h3>
          <div className="answer-box">
            <h4>Question:</h4>
            <p className="question-text">{question}</p>
            <h4>Answer:</h4>
            <p className="answer-text">{answer}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default VQAInterface;