import {  Routes , Route } from "react-router-dom";
import { lazy , Suspense } from "react";
import "./App.css";
import NavBar from "./components/Nav";
const Home = lazy(() => import('./pages/home'));
const About = lazy(() => import('./pages/about'));
const Contact = lazy(() => import('./pages/contact'));
const Display = lazy(() => import('./pages/display'));

const App = () => {

  return (
        <>
        {/* <NavBar /> */}
        <Suspense fallback={<div className="container">Loading...</div>}>
        <Routes>
          <Route path="/" element={<Display />} />
          {/* <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/test" element={<Test />} /> */}
          <Route path="*" element={<h1>404 Not Found</h1>} />
        </Routes>
        </Suspense>
        </>
  );
}

export default App;
