import { streamText, convertToModelMessages, UIMessage } from 'ai'

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json()

  const result = streamText({
    model: 'openai/gpt-4o-mini',
    system: `You are a specialized AI assistant for transcriptome and cell biology analysis. 
You help researchers understand:
- Gene expression patterns and regulation
- Cell state transitions and differentiation
- Transcriptomic data interpretation
- Cellular processes like transcription, translation, and post-transcriptional modifications
- Single-cell RNA sequencing analysis
- Cell cluster identification and annotation

Provide concise, scientifically accurate responses. When discussing specific genes or pathways, include relevant biological context.
Answer in the same language as the user's question.`,
    messages: await convertToModelMessages(messages),
  })

  return result.toUIMessageStreamResponse()
}
