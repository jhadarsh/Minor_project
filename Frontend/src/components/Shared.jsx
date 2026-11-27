import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'

export const inputCls =
  'w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 shadow-sm outline-none transition focus:border-blue-400 focus:ring-4 focus:ring-blue-100 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:opacity-60'
export const labelCls = 'grid gap-1 text-xs font-semibold text-slate-600'

const btnBase =
  'inline-flex items-center justify-center gap-1.5 rounded-xl px-3.5 py-2 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50'
const variants = {
  primary: 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-md shadow-blue-500/25 hover:from-blue-600 hover:to-indigo-700',
  secondary: 'border border-blue-200 bg-blue-50 text-blue-800 hover:bg-blue-100',
  stop: 'border border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100',
}

export function Button({ variant = 'secondary', className = '', disabled, type, ...props }) {
  return (
    <motion.button
      type={type}
      disabled={disabled}
      whileHover={disabled ? undefined : { y: -1.5 }}
      whileTap={disabled ? undefined : { scale: 0.96 }}
      transition={{ type: 'spring', stiffness: 400, damping: 20 }}
      className={`${btnBase} ${variants[variant]} ${className}`}
      {...props}
    />
  )
}

export function Card({ className = '', delay = 0, children, ...props }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
      className={`min-h-0 rounded-2xl border border-white bg-white/80 p-4 shadow-xl shadow-slate-300/30 backdrop-blur-md ${className}`}
      {...props}
    >
      {children}
    </motion.section>
  )
}

export function Lbl({ icon: Icon, children }) {
  return (
    <span className="flex items-center gap-1.5">
      <Icon size={13} className="text-blue-600" />
      {children}
    </span>
  )
}

export function Title({ icon: Icon, children }) {
  return (
    <h2 className="mb-3 flex items-center gap-2 text-base font-bold text-slate-800">
      <span className="grid h-7 w-7 place-items-center rounded-lg bg-gradient-to-br from-blue-100 to-indigo-100 text-blue-700"><Icon size={15} /></span>
      {children}
    </h2>
  )
}

const bars = [30, 55, 85, 100, 70, 45, 28]

export function Wave({ className = '' }) {
  return (
    <span className={`inline-flex h-4 w-5 items-center justify-center gap-[2px] ${className}`} aria-hidden="true">
      {bars.map((h, i) => (
        <motion.i
          key={i}
          className="block w-[2.5px] rounded-full bg-current"
          animate={{ height: [`${h}%`, `${Math.max(25, 110 - h)}%`, `${h}%`] }}
          transition={{ duration: 1.1 + i * 0.08, repeat: Infinity, ease: 'easeInOut' }}
        />
      ))}
    </span>
  )
}

export function Notice({ children }) {
  return (
    <AnimatePresence>
      {children ? (
        <motion.p
          key="notice"
          role="alert"
          initial={{ opacity: 0, height: 0, y: -6 }}
          animate={{ opacity: 1, height: 'auto', y: 0 }}
          exit={{ opacity: 0, height: 0 }}
          className="flex items-start gap-1.5 overflow-hidden rounded-xl border border-rose-100 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700"
        >
          <AlertTriangle size={14} className="mt-px shrink-0" />
          <span>{children}</span>
        </motion.p>
      ) : null}
    </AnimatePresence>
  )
}