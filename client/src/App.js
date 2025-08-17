import React, { useState } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import VQAInterface from './components/VQAInterface';
import Footer from './components/Footer';

function App() {
  return (
    <div className="App">
      <Navbar />
      <main className="main-content">
        <h1>Medical VQA System</h1>
        <p>Upload an image, select a model, and ask a question to analyze medical images</p>
        <VQAInterface />
      </main>
      <Footer />
    </div>
  );
}

export default App;