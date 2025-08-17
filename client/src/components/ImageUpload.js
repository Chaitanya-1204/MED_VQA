import React, { useState, useRef } from 'react';
import './ImageUpload.css';

function ImageUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
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
    } else {
      alert('Please select an image file');
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      alert('Please select an image first');
      return;
    }

    setIsUploading(true);
    
    // Create form data for file upload
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      // Replace with your actual API endpoint
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        alert('Image uploaded successfully!');
        // You can handle the response data here
        console.log(data);
      } else {
        alert('Upload failed');
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Error uploading image. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const clearImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
  };

  return (
    <div className="image-upload-container">
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
            <p className="dropzone-text">Drag & Drop your image here or click to browse</p>
            <p className="dropzone-hint">Supports: JPG, PNG, GIF</p>
          </div>
        )}
      </div>

      <button 
        className={`submit-button ${!selectedFile ? 'disabled' : ''} ${isUploading ? 'uploading' : ''}`}
        onClick={handleSubmit}
        disabled={!selectedFile || isUploading}
      >
        {isUploading ? 'Uploading...' : 'Submit Image'}
      </button>
    </div>
  );
}

export default ImageUpload;