import axios from 'axios';

// Configure axios instance
const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    let message = 'An error occurred';
    
    if (error.response) {
      message = error.response.data?.detail || error.response.statusText;
    } else if (error.request) {
      message = 'No response from server. Please check your connection.';
    } else {
      message = error.message;
    }
    
    throw new Error(message);
  }
);

export const detectBacteria = async (formData) => {
  try {
    const response = await api.post('/api/v1/detect', formData, {
      params: {
        confidence_threshold: 0.5,
        top_k: 3
      }
    });
    return response;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const batchDetect = async (files) => {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  
  return await api.post('/api/v1/detect/batch', formData);
};

export const healthCheck = async () => {
  return await api.get('/health');
};

export default api;
