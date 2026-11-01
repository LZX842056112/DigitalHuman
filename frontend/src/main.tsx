import React from 'react';
import {createRoot} from 'react-dom/client';
import PhotoApp from './photo-main';

createRoot(document.getElementById('root')!).render(<React.StrictMode><PhotoApp /></React.StrictMode>);
