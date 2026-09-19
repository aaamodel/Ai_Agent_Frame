import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router-dom";

import { AppShell } from "@/components/layout/AppShell";
import { ChatPage } from "@/features/chat/ChatPage";
import { DocumentsPage } from "@/features/documents/DocumentsPage";
import { KnowledgeBasePage } from "@/features/knowledgebase/KnowledgeBasePage";

const queryClient = new QueryClient();

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <ChatPage /> },
      { path: "documents", element: <DocumentsPage /> },
      { path: "documents/:collectionName", element: <DocumentsPage /> },
      { path: "knowledgebase", element: <KnowledgeBasePage /> },
      { path: "knowledgebase/:collectionName", element: <KnowledgeBasePage /> },
    ],
  },
]);

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
