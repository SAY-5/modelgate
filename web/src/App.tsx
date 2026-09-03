import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { InputChecks } from "./components/InputChecks";
import { MetricsPanel } from "./components/MetricsPanel";
import { Nav } from "./components/Nav";
import { ShadowRuns } from "./components/ShadowRuns";
import { ZeroDropSwap } from "./components/ZeroDropSwap";
import { ServiceProvider } from "./state/ServiceProvider";
import "./styles/sections.css";

export default function App() {
  return (
    <ServiceProvider>
      <div className="atmosphere" aria-hidden="true" />
      <div className="grain" aria-hidden="true" />
      <div className="page">
        <a href="#checks" className="sr-only">
          Skip to content
        </a>
        <Nav />
        <main>
          <Hero />
          <InputChecks />
          <ShadowRuns />
          <ZeroDropSwap />
          <MetricsPanel />
        </main>
        <Footer />
      </div>
    </ServiceProvider>
  );
}
