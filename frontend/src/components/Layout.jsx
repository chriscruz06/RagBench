import { Outlet } from 'react-router-dom'
import Header from './Header'

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col bg-cream">
      <Header />
      <main className="flex-1 flex justify-center px-6 py-24">
        <div className="w-full max-w-content">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
