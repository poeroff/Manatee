import { ChatPanel } from '@/components/chat-panel'
import { CellVisualization } from '@/components/cell-visualization'

export default function Home() {
  return (
    <main className="flex h-screen w-screen overflow-hidden bg-background">
      {/* Left: Chat Panel - compact width */}
      <aside className="h-full w-80 shrink-0 border-r border-border">
        <ChatPanel />
      </aside>

      {/* Right: Cell Visualization - takes remaining space */}
      <section className="flex-1">
        <CellVisualization />
      </section>
    </main>
  )
}
