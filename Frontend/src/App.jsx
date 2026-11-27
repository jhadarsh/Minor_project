import { useEffect, useState } from 'react'
import { motion, MotionConfig } from 'framer-motion'
import { MessageCircle, Settings } from 'lucide-react'
import AdminPage from './components/AdminPage'
import QuestionPage from './components/QuestionPage'
import { Wave } from './components/Shared'

function getPage() {
  return window.location.pathname.replace(/\/$/, '') === '/admin' ? '/admin' : '/ask'
}

const tabs = [
  { path: '/ask', label: 'Ask', icon: MessageCircle },
  { path: '/admin', label: 'Admin', icon: Settings },
]

const pageAnim = (visible) => ({
  animate: visible ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 },
  transition: { duration: 0.35, ease: 'easeOut' },
})

export default function App() {
  const [page, setPage] = useState(getPage)
  const [queryKey, setQueryKey] = useState('')

  const [scrolled, setScrolled] = useState(false)

useEffect(() => {
  const onScroll = () => setScrolled(window.scrollY > 8)
  onScroll()
  window.addEventListener('scroll', onScroll, { passive: true })
  return () => window.removeEventListener('scroll', onScroll)
}, [])

  useEffect(() => {
    const updatePage = () => setPage(getPage())
    window.addEventListener('popstate', updatePage)
    return () => window.removeEventListener('popstate', updatePage)
  }, [])

  function navigate(path) {
    if (window.location.pathname !== path) window.history.pushState({}, '', path)
    setPage(path)
  }

  function follow(event, path) {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return
    event.preventDefault()
    navigate(path)
  }

  return (
    <MotionConfig reducedMotion="user">
      <div className="relative flex h-dvh flex-col overflow-x-hidden overflow-y-auto bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 text-slate-800 lg:overflow-y-hidden [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
        {/* soft floating background glows */}
        <motion.div aria-hidden className="pointer-events-none absolute -left-24 -top-24 h-72 w-72 rounded-full bg-blue-300/40 blur-3xl"
          animate={{ x: [0, 40, 0], y: [0, 30, 0] }} transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }} />
        <motion.div aria-hidden className="pointer-events-none absolute -bottom-28 -right-20 h-80 w-80 rounded-full bg-indigo-300/40 blur-3xl"
          animate={{ x: [0, -40, 0], y: [0, -30, 0] }} transition={{ duration: 16, repeat: Infinity, ease: 'easeInOut' }} />

       <header
  className={`sticky top-0 z-30 shrink-0 transition-all duration-300 ${
    scrolled
      ? 'bg-white/65 backdrop-blur-xl border-b border-white/60 shadow-sm shadow-blue-900/5'
      : 'bg-transparent border-b border-transparent'
  }`}
>
  <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between gap-3 px-4">
    <a
      href="/ask"
      onClick={(e) => follow(e, '/ask')}
      className="flex items-center gap-2.5 font-bold text-slate-900 no-underline"
    >
      <motion.span
        whileHover={{ rotate: -8, scale: 1.08 }}
        className="grid h-8 w-8 place-items-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-sm shadow-blue-600/20"
      >
        <Wave />
      </motion.span>
      <span className="text-[15px]">Voice Q&amp;A</span>
    </a>

    <nav
      aria-label="Pages"
      className="relative flex gap-1 rounded-xl bg-white/70 p-1 ring-1 ring-slate-200/70 backdrop-blur-sm"
    >
      {tabs.map((tab) => {
        const current = page === tab.path
        return (
          <a
            key={tab.path}
            href={tab.path}
            onClick={(e) => follow(e, tab.path)}
            aria-current={current ? 'page' : undefined}
            className={`relative rounded-lg px-3 py-1 text-sm font-semibold no-underline transition-colors ${
              current ? 'text-blue-600' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            {current && (
              <motion.span
                layoutId="tab-pill"
                className="absolute inset-0 rounded-lg bg-white shadow-sm ring-1 ring-slate-200/60"
                transition={{ type: 'spring', stiffness: 400, damping: 32 }}
              />
            )}
            <span className="relative flex items-center gap-1.5">
              <tab.icon size={14} />
              {tab.label}
            </span>
          </a>
        )
      })}
    </nav>
  </div>
</header>

        <main className="relative z-10 mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 py-3 lg:min-h-0">
          <div hidden={page !== '/ask'} className="flex-1 lg:min-h-0">
            <motion.div className="h-full" {...pageAnim(page === '/ask')}>
              <QuestionPage active={page === '/ask'} queryKey={queryKey} setQueryKey={setQueryKey} />
            </motion.div>
          </div>
          <div hidden={page !== '/admin'} className="flex-1 lg:min-h-0">
            <motion.div className="h-full" {...pageAnim(page === '/admin')}>
              <AdminPage active={page === '/admin'} onTry={(key) => { setQueryKey(key); navigate('/ask') }} />
            </motion.div>
          </div>
        </main>
      </div>
    </MotionConfig>
  )
}