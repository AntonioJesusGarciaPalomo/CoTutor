import { useState } from 'react';
import { SolverInput } from './components/SolverInput';
import { SolutionView, type SolutionData } from './components/SolutionView';
import { TutorChat } from './components/TutorChat';
import { api } from './services/api';
import './index.css';

interface ChatMessage {
  role: 'user' | 'tutor';
  content: string;
}

function App() {
  const [solution, setSolution] = useState<SolutionData | null>(null);
  const [isSolving, setIsSolving] = useState(false);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isStartingSession, setIsStartingSession] = useState(false);

  const handleSolve = async (problemText: string) => {
    setIsSolving(true);
    setSolution(null);
    setSessionId(null);
    setChatMessages([]);

    try {
      const response = await api.post('/solver/solve', { problem_text: problemText });
      setSolution({
        ...response.data,
        problem_text: problemText
      });
    } catch (error) {
      console.error("Failed to solve:", error);
      alert("Error al conectar con el Solver Agent");
    } finally {
      setIsSolving(false);
    }
  };

  const handleStartSession = async () => {
    if (!solution) return;
    setIsStartingSession(true);

    try {
      const response = await api.post('/tutor/session', {
        problem_text: solution.problem_text,
        solution: solution
      });
      setSessionId(response.data.session_id);

      // Initial greeting from tutor (simulated or fetched if backend provided)
      setChatMessages([{ role: 'tutor', content: "¡Hola! He analizado tu problema. ¿Cómo te gustaría empezar?" }]);
    } catch (error) {
      console.error("Failed to start session:", error);
      alert("Error al iniciar sesión de tutoría");
    } finally {
      setIsStartingSession(false);
    }
  };

  const handleSendMessage = async (text: string) => {
    if (!sessionId) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    setChatMessages(prev => [...prev, userMsg]);
    setIsChatLoading(true);

    try {
      const response = await api.post('/tutor/chat', {
        session_id: sessionId,
        message: text
      });

      const tutorMsg: ChatMessage = { role: 'tutor', content: response.data.content };
      setChatMessages(prev => [...prev, tutorMsg]);
    } catch (error) {
      console.error("Failed to send message:", error);
      alert("Error al enviar mensaje");
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8 font-sans text-slate-900">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-extrabold tracking-tight text-indigo-900 sm:text-5xl mb-4">
            CoTutor AI
          </h1>
          <p className="text-lg text-slate-600">
            Tu asistente inteligente para resolver y entender matemáticas.
          </p>
        </div>

        <SolverInput onSolve={handleSolve} isLoading={isSolving} />

        {solution && (
          <div className="animate-fade-in-up">
            <SolutionView
              solution={solution}
              onStartTutor={handleStartSession}
              isStartingSession={isStartingSession}
            />
          </div>
        )}

        {sessionId && (
          <div className="animate-fade-in-up">
            <TutorChat
              messages={chatMessages}
              onSendMessage={handleSendMessage}
              isLoading={isChatLoading}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
