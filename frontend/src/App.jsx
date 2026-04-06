import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Study from './pages/Study'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Study />} />
      </Route>
    </Routes>
  )
}
