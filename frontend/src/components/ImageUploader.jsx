import React, { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';

function ImageUploader({ onUpload }) {
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFile = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Trigger upload
      setTimeout(() => onUpload(file), 100);
    }
  }, [onUpload]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleChange = useCallback((e) => {
    const file = e.target.files[0];
    handleFile(file);
  }, [handleFile]);

  return (
    <div className="uploader-container">
      <div className="upload-card">
        <h2 className="card-title">Upload Image</h2>
        
        <div
          className={`dropzone ${dragActive ? 'dropzone-active' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            type="file"
            id="file-upload"
            className="file-input"
            accept="image/*"
            onChange={handleChange}
          />
          <label htmlFor="file-upload" className="dropzone-label">
            <Upload className="upload-icon" />
            <p className="dropzone-text-main">
              {dragActive ? 'Drop the image here' : 'Drag & drop an image here'}
            </p>
            <p className="dropzone-text-sub">or click to select a file</p>
            <p className="dropzone-text-formats">Supported formats: JPEG, PNG, GIF, BMP</p>
            <button type="button" className="select-button">
              <ImageIcon className="button-icon" />
              Select Image
            </button>
          </label>
        </div>

        {preview && (
          <div className="preview-container">
            <p className="preview-label">Image Preview:</p>
            <img src={preview} alt="Preview" className="preview-image" />
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageUploader;