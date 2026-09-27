import { Routes, Route } from "react-router-dom";
import Display from "./pages/display";

const App = () => (
  <Routes>
    <Route path="/" element={<Display />} />
    <Route path="*" element={<h1>404 Not Found</h1>} />
  </Routes>
);

export default App;
