import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { LoadProvider } from "./components/LoadProvider";
import "./styles/base.css";
import TrackView from "./views/TrackView";
import HeatsView from "./views/HeatsView";
import LimitsView from "./views/LimitsView";
import PapersView from "./views/PapersView";
import SpecimenView from "./views/SpecimenView";

export default function App() {
  return (
    <LoadProvider>
      <BrowserRouter>
        <div className="app-shell">
          <nav className="rail">
            <div className="wordmark">relay-lane</div>
            <NavLink to="/">Track</NavLink>
            <NavLink to="/heats">Heats</NavLink>
            <NavLink to="/limits">Limits</NavLink>
            <NavLink to="/papers">Papers</NavLink>
            {import.meta.env.DEV && <NavLink to="/specimen">Specimen</NavLink>}
          </nav>
          <main className="main">
            <Routes>
              <Route path="/" element={<TrackView />} />
              <Route path="/heats" element={<HeatsView />} />
              <Route path="/limits" element={<LimitsView />} />
              <Route path="/papers" element={<PapersView />} />
              <Route path="/specimen" element={<SpecimenView />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </LoadProvider>
  );
}
