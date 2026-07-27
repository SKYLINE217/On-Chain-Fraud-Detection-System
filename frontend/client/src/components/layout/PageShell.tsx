import { Outlet } from 'react-router-dom';
import NavBar from './NavBar';
import DisclaimerBanner from '../shared/DisclaimerBanner';

export default function PageShell() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col relative">
      <NavBar />
      <main className="flex-1 w-full max-w-[1440px] mx-auto px-4 md:px-12 pt-8 pb-24">
        <Outlet />
      </main>
      <DisclaimerBanner />
    </div>
  );
}
