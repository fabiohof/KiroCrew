/**
 * The agent rail shows a role's crew by DISPLAY NAME, never by the stored
 * `member_id` -- not even for a frame.
 *
 * A role added from the picker stores the crew's `member_id`
 * (`crew-program-manager`). The rail resolves that through the roster
 * (`agentDisplayLabel`), but the roster is a second fetch: before it answers,
 * the only value the rail could render is the raw id, which is machine text
 * where a name belongs and would flash on every visit. The subtitle is held
 * until the catalog has settled, then shows the name.
 */
import { describe, it, expect, vi, beforeEach, beforeAll } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChannelPage from '../pages/ChannelPage'
import { renderWithProviders } from './helpers'
import { api } from '../api/client'

vi.mock('../api/client')

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

type Raw = Record<string, unknown>

const channel: Raw = {
  id: 'ch1', topic: 'Gamma rollout', messages: [],
  members: {
    a1: {
      id: 'a1', role: 'Researcher', agent_name: 'crew-program-manager',
      state: 'listening', listen_mode: 'mention', approval_policy: 'writes',
    },
  },
}

const ROSTER = [
  { name: 'default', member_id: 'default', display_name: '', kiro_agent: 'kirocrew', workspace: 'default', memory_store: 'default', description: '', source: 'kirocrew' },
  { name: 'Crew Program Manager', member_id: 'crew-program-manager', display_name: 'Crew Program Manager', kiro_agent: 'kirocrew', workspace: 'default', memory_store: 'member-crew-program-manager', description: '', source: 'kirocrew' },
]

let answerCatalog: (value: { agents: typeof ROSTER; default_agent: string }) => void = () => {}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(api).channelsList = vi.fn().mockResolvedValue({ channels: [channel] })
  vi.mocked(api).channelGet = vi.fn().mockResolvedValue(channel)
  vi.mocked(api).channelPresets = vi.fn().mockResolvedValue({})
  vi.mocked(api).kirocrewAgents = vi.fn().mockResolvedValue({ agents: ROSTER, default_agent: 'default' })
  // The roster fetch the rail waits on, answered by the test.
  vi.mocked(api).agentCatalog = vi.fn().mockReturnValue(
    new Promise(resolve => { answerCatalog = resolve }),
  )
})

describe('ChannelPage agent rail crew label', () => {
  it('holds the subtitle until the roster answers, then shows the display name', async () => {
    renderWithProviders(<ChannelPage />)
    await waitFor(() => expect(screen.queryByText('Loading channels...')).not.toBeInTheDocument())
    await userEvent.click(await screen.findByRole('button', { name: '1 agent' }))
    expect(screen.getByText('Researcher')).toBeInTheDocument()

    // Roster still loading: the stored id is not on the page.
    expect(screen.queryByText('crew-program-manager')).not.toBeInTheDocument()
    expect(screen.queryByText('Crew Program Manager')).not.toBeInTheDocument()

    answerCatalog({ agents: ROSTER, default_agent: 'default' })

    await waitFor(() => expect(screen.getByText('Crew Program Manager')).toBeInTheDocument())
    expect(screen.queryByText('crew-program-manager')).not.toBeInTheDocument()
  })
})
