import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import UniversalRestoration from './pages/UniversalRestoration';
import HardRouted from './pages/HardRouted';
import SoftMoE from './pages/SoftMoE';
import FaceToSketch from './pages/FaceToSketch';
import SystemModels from './pages/SystemModels';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<UniversalRestoration />} />
          <Route path="hard-routed" element={<HardRouted />} />
          <Route path="soft-moe" element={<SoftMoE />} />
          <Route path="face-sketch" element={<FaceToSketch />} />
          <Route path="system" element={<SystemModels />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
