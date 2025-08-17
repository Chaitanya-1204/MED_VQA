import React, { useState } from 'react';
import './Navbar.css';

function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <h2>Techno Titans</h2>
        </div>
        
        <div className="menu-icon" onClick={toggleMenu}>
          <span className={isMenuOpen ? 'menu-icon-bar open' : 'menu-icon-bar'}></span>
          <span className={isMenuOpen ? 'menu-icon-bar open' : 'menu-icon-bar'}></span>
          <span className={isMenuOpen ? 'menu-icon-bar open' : 'menu-icon-bar'}></span>
        </div>
        
        <ul className={isMenuOpen ? 'nav-menu active' : 'nav-menu'}>
          <li className="nav-item">
            <a href="#home" className="nav-link">Home</a>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;